"""Regression coverage for the copied decomposed GDN entry point."""

import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.standards import DEFAULT_STANDARDS


def recurrent_reference(
    q, k, v, g, beta, initial_state, cu_seqlens, scale, normalize, chunk_size=64
):
    """Independent tokenwise recurrence, including entering chunk states."""
    dtype, device = q.dtype, q.device
    q, k, v, g, beta = (x.detach().cpu() for x in (q, k, v, g, beta))
    if normalize:
        q, k = (
            (
                x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            for x in (q, k)
        )
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    batch, tokens, groups, key_dim = k.shape
    heads, value_dim = v.shape[-2:]
    q, k = (x.repeat_interleave(heads // groups, dim=2) for x in (q, k))
    scale = key_dim**-0.5 if scale is None else scale
    if cu_seqlens is None:
        sequences = [(b, 0, tokens) for b in range(batch)]
    else:
        bounds = cu_seqlens.cpu().tolist()
        sequences = [(0, start, end) for start, end in zip(bounds, bounds[1:])]
    output = torch.empty_like(v)
    final_states, chunks = [], []
    for sequence, (b, start, end) in enumerate(sequences):
        state = (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float64, device="cpu")
            if initial_state is None
            else initial_state[sequence].detach().cpu().double().clone()
        )
        for t in range(start, end):
            if (t - start) % chunk_size == 0:
                chunks.append(state.to(dtype))
            state = state * g[b, t].exp()[:, None, None]
            residual = v[b, t] - torch.einsum("hk,hkv->hv", k[b, t], state)
            state = state + torch.einsum(
                "hk,hv->hkv", k[b, t], residual * beta[b, t, :, None]
            )
            output[b, t] = torch.einsum("hk,hkv->hv", q[b, t], state) * scale
        final_states.append(state.float())
    h = torch.stack(chunks).reshape(batch, -1, heads, key_dim, value_dim)
    return (
        output.to(device=device, dtype=dtype),
        torch.stack(final_states).to(device),
        h.to(device),
    )


def generate_data(bounds, dtype, with_state=True, noncontiguous=False):
    # Synthetic token counts with Qwen3-Next TP4 head dimensions (Hg=4/H=8,
    # K=V=128), following the existing chunk_o public tests.
    torch.manual_seed(42)
    tokens = 65 if bounds is None else bounds[-1]
    groups, heads, dim = 4, 8, 128
    q = torch.randn(1, tokens, groups, dim, dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, tokens, heads, dim, dtype=dtype) * 0.1
    g = -torch.rand(1, tokens, heads, dtype=torch.float32) * 0.05
    beta = torch.rand(1, tokens, heads, dtype=dtype)
    data = dict(q=q, k=k, v=v, g=g, beta=beta)
    if noncontiguous:
        for name, tensor in data.items():
            backing = torch.empty(
                *tensor.shape[:-1], tensor.shape[-1] * 2, dtype=tensor.dtype
            )
            data[name] = backing[..., ::2].copy_(tensor)
    data["initial_state"] = (
        torch.randn(1 if bounds is None else len(bounds) - 1, heads, dim, dim) * 0.01
        if with_state
        else None
    )
    data["cu_seqlens"] = (
        None if bounds is None else torch.tensor(bounds, dtype=torch.int32)
    )
    return data


@pytest.mark.parametrize("impl", ["auto", "triton"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "bounds,with_state,normalize,output_final_state,scale,noncontiguous",
    [
        pytest.param(None, True, False, True, None, False, id="fixed-chunk-tail"),
        pytest.param((0, 1), False, True, False, 0.25, False, id="one-token"),
        pytest.param((0, 64), True, True, True, None, False, id="full-chunk"),
        pytest.param(
            (0, 1, 66, 131),
            True,
            True,
            False,
            0.0,
            True,
            id="packed-grouped-strided-zero-scale",
        ),
        pytest.param(
            (0, 31, 100),
            False,
            False,
            True,
            None,
            False,
            id="packed-without-initial-state",
        ),
    ],
)
def test_chunk_gated_delta_rule_npu(
    impl,
    dtype,
    bounds,
    with_state,
    normalize,
    output_final_state,
    scale,
    noncontiguous,
    monkeypatch,
):
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu as legacy
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    monkeypatch.setenv("GDN_USE_MEGA_GDN", "0")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    data = generate_data(bounds, dtype, with_state, noncontiguous)
    saved = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    options = dict(use_qk_l2norm_in_kernel=normalize, scale=scale)
    # Omit the flag in True cases to check the legacy True default.
    if not output_final_state:
        options["output_final_state"] = False
    actual = chunk_gated_delta_rule_npu(**data, **options, impl=impl)
    expected_legacy = legacy(**data, **options)
    assert_accuracy(actual, expected_legacy)
    reference = list(
        recurrent_reference(
            **data,
            scale=scale,
            normalize=normalize,
        )
    )
    if not output_final_state:
        reference[1] = None
    # FP32 final states inherit the pipeline's input-dtype rounding error.
    standard = {"torch.float32": DEFAULT_STANDARDS[str(dtype)]}
    assert_accuracy(actual, tuple(reference), standard=standard)
    for name, original in saved.items():
        torch.testing.assert_close(data[name], original, atol=0, rtol=0)


@pytest.mark.parametrize("output_final_state", [False, True])
def test_chunk_gated_delta_rule_npu_mega(output_final_state, monkeypatch):
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu as legacy
    from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("The legacy mega_chunk_gdn native operator is required")
    monkeypatch.setenv("GDN_USE_MEGA_GDN", "1")
    data = generate_data((0, 31, 100), torch.float16)
    options = dict(output_final_state=output_final_state)
    assert_accuracy(
        chunk_gated_delta_rule_npu(**data, **options), legacy(**data, **options)
    )
