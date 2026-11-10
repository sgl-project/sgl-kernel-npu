"""Check the complete migrated chunk pipeline against tokenwise recurrence.

The oracle shares no WY, triangular inverse or chunk-state implementation with
the DUT. It catches reordered gates, lost sequence boundaries and state-layout
changes that independent leaf tests cannot detect.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.common import get_test_impls, require_test_impl


def recurrent_reference(q, k, v, g, beta, state, offsets, scale, normalize, chunk_size):
    device = q.device
    dtype = q.dtype
    q, k, v, g, beta = (x.detach().cpu().double() for x in (q, k, v, g, beta))
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    batch, length, groups, key_dim = k.shape
    heads, value_dim = v.shape[-2:]
    if offsets is None:
        sequences = [(b, 0, length) for b in range(batch)]
        chunks = (length + chunk_size - 1) // chunk_size
    else:
        boundaries = offsets.cpu().tolist()
        sequences = [(0, start, end) for start, end in zip(boundaries, boundaries[1:])]
        chunks = sum(
            (end - start + chunk_size - 1) // chunk_size for _, start, end in sequences
        )
    out = torch.empty_like(v)
    h = torch.empty(batch, chunks, heads, key_dim, value_dim, dtype=torch.float64)
    final = torch.empty(len(sequences), heads, key_dim, value_dim, dtype=torch.float64)
    packed_chunk = 0
    effective_scale = key_dim**-0.5 if scale is None else scale
    for index, (b, start, end) in enumerate(sequences):
        current = (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float64)
            if state is None
            else state[index].detach().cpu().double().clone()
        )
        for pos in range(start, end):
            if (pos - start) % chunk_size == 0:
                chunk = (
                    packed_chunk if offsets is not None else (pos - start) // chunk_size
                )
                h[b, chunk] = current
                packed_chunk += 1
            keys = k[b, pos].repeat_interleave(heads // groups, dim=0)
            queries = q[b, pos].repeat_interleave(heads // groups, dim=0)
            current = current * g[b, pos].exp()[:, None, None]
            prediction = torch.einsum("hk,hkv->hv", keys, current)
            delta = beta[b, pos, :, None] * (v[b, pos] - prediction)
            current = current + keys[:, :, None] * delta[:, None, :]
            out[b, pos] = torch.einsum("hk,hkv->hv", queries, current) * effective_scale
        final[index] = current
    return (
        out.to(device=device, dtype=dtype),
        final.to(device=device, dtype=torch.float32),
        h.to(device=device, dtype=dtype),
    )


def generate_inputs(mode, dtype, with_state, noncontiguous=False):
    # Borrow Qwen3-Next TP dimensions Hg=4/H=8, K=V=128; synthetic short
    # sequences exercise tails and chunk restarts, rather than a model trace.
    batch, length, groups, heads, dim = 1, 129, 4, 8, 128
    q = torch.randn(batch, length, groups, dim, device="npu", dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(batch, length, heads, dim, device="npu", dtype=dtype) * 0.1
    g = -torch.rand(batch, length, heads, device="npu", dtype=torch.float32) * 0.05
    beta = torch.rand(batch, length, heads, device="npu", dtype=dtype)
    offsets = (
        torch.tensor([0, 31, length], device="npu", dtype=torch.int32)
        if mode == "packed"
        else None
    )
    num_sequences = 2 if offsets is not None else batch
    state = (
        torch.randn(num_sequences, heads, dim, dim, device="npu", dtype=torch.float32)
        * 0.01
        if with_state
        else None
    )
    if noncontiguous:

        def strided(x):
            storage = torch.empty(
                *x.shape[:-1], x.shape[-1] * 2, device=x.device, dtype=x.dtype
            )
            view = storage[..., ::2]
            view.copy_(x)
            return view

        q, k, v, g, beta = map(strided, (q, k, v, g, beta))
    return dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=state, cu_seqlens=offsets)


def compare_results(actual, expected, output_final_state):
    assert len(actual) == 3
    for index in (0, 2):
        torch.testing.assert_close(actual[index], expected[index], atol=3e-3, rtol=5e-2)
    if output_final_state:
        torch.testing.assert_close(actual[1], expected[1], atol=3e-3, rtol=5e-2)
    else:
        assert actual[1] is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["packed", "fixed-b1"])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_chunk_gated_delta_rule_npu(dtype, mode, with_state, normalize, monkeypatch):
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    monkeypatch.setenv("GDN_USE_MEGA_GDN", "0")
    data = generate_inputs(mode, dtype, with_state, noncontiguous=normalize)
    snapshots = {
        name: value.clone()
        for name, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    expected = recurrent_reference(
        data["q"],
        data["k"],
        data["v"],
        data["g"],
        data["beta"],
        data["initial_state"],
        data["cu_seqlens"],
        None,
        normalize,
        64,
    )
    impl, _ = get_test_impls()
    require_test_impl(chunk_gated_delta_rule_npu, impl)
    # Omit output_final_state to guard the original wrapper's True default.
    actual = chunk_gated_delta_rule_npu(
        **data, use_qk_l2norm_in_kernel=normalize, impl=impl
    )
    compare_results(actual, expected, True)
    for name, before in snapshots.items():
        assert torch.equal(data[name], before), f"input mutated: {name}"
    # Explicit scale zero only zeros the attention output; states still update.
    zero = chunk_gated_delta_rule_npu(
        **data,
        scale=0.0,
        output_final_state=False,
        use_qk_l2norm_in_kernel=normalize,
        impl=impl,
    )
    assert torch.count_nonzero(zero[0]) == 0
    assert zero[1] is None
    torch.testing.assert_close(zero[2], actual[2], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_gated_delta_rule_npu_mega(dtype, monkeypatch):
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("The optional mega_chunk_gdn native operator is not installed")
    monkeypatch.setenv("GDN_USE_MEGA_GDN", "1")
    data = generate_inputs("packed", dtype, True)
    # The existing mega backend stores FP16 intermediates and uses chunks of 128.
    # Normalize inputs before entry to keep both backend distributions stable.
    for name in ("q", "k"):
        data[name] = torch.nn.functional.normalize(data[name], dim=-1)
    expected = recurrent_reference(
        data["q"],
        data["k"],
        data["v"],
        data["g"],
        data["beta"],
        data["initial_state"],
        data["cu_seqlens"],
        None,
        False,
        128,
    )
    impl, _ = get_test_impls()
    require_test_impl(chunk_gated_delta_rule_npu, impl)
    compare_results(chunk_gated_delta_rule_npu(**data, impl=impl), expected, True)
