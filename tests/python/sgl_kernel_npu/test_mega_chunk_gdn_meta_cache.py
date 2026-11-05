"""Tests for the meta-cache / workspace-pool paths in ``run_mega_chunk_gdn``.

The optimized wrapper keeps a grow-only token slab, content-keyed cu_seqlens
caches and per-shape scratch buffers so that repeated prefill calls (across
GDN layers and across iterations) do not re-allocate or re-initialize
intermediates. These tests pin the contracts that make those caches safe:

* slab grow/shrink across calls keeps outputs correct (slab reuse + slicing);
* toggling the meta cache at runtime does not change numerics;
* repeated calls with identical ``cu_seqlens`` content (fresh tensor objects
  each call, i.e. the content-keyed D2H cache path) stay correct;
* ``return_internals=False`` keeps the documented dtype/shape contract;
* ``clear_mega_workspace_cache()`` does not break subsequent calls.

A regression gate is included but gated behind ``SGLANG_NPU_RUN_META_CACHE_PERF=1``
because it is timing sensitive: it re-runs the same AscendC kernel under the
upstream per-call allocation pattern and the cached wrapper and asserts the
cached path is not slower end-to-end. Run it manually before merging, not on
every CI cycle.
"""

import os
import time

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_native
from sgl_kernel_npu.fla.mega_chunk_gdn import (
    clear_mega_workspace_cache,
    run_mega_chunk_gdn,
)
from sgl_kernel_npu.fla.utils import is_gdn_meta_cache_enabled, set_gdn_meta_cache


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")

CHUNK_SIZE = 128


def _op_registered() -> bool:
    return hasattr(torch.ops.npu, "mega_chunk_gdn")


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float().cpu() - expected.float()).abs()
    max_abs = diff.max().item()
    if max_abs <= 1e-2:
        return

    rmse = torch.sqrt((diff.flatten() ** 2).mean()).item()
    base = torch.sqrt((expected.float().flatten() ** 2).mean()).item()
    ratio = rmse / max(base, 1e-8)
    assert ratio < 0.05, f"{name} max_abs={max_abs:.6f} rmse_ratio={ratio:.6f}"


def _native_reference(
    q, k, v, g, beta, cu_seqlens, initial_state=None, output_final_state=True
):
    if q.shape[2] != v.shape[2]:
        assert v.shape[2] % q.shape[2] == 0
        group_size = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(group_size, dim=2)
        k = k.repeat_interleave(group_size, dim=2)

    outs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        cur_initial_state = (
            None if initial_state is None else initial_state[seq_idx : seq_idx + 1]
        )
        out, final_state = chunk_gated_delta_rule_native(
            query=q[:, start:end],
            key=k[:, start:end],
            value=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            chunk_size=CHUNK_SIZE,
            initial_state=cur_initial_state,
            output_final_state=output_final_state,
        )
        outs.append(out)
        if output_final_state:
            final_states.append(final_state)
    return torch.cat(outs, dim=1), (
        torch.cat(final_states, dim=0) if output_final_state else None
    )


def _make_inputs(total_tokens, cu_list, H, Hg, D, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(
        torch.randn(1, total_tokens, Hg, D, generator=gen), p=2, dim=-1
    ).to(torch.float16)
    k = F.normalize(
        torch.randn(1, total_tokens, Hg, D, generator=gen), p=2, dim=-1
    ).to(torch.float16)
    v = torch.randn(1, total_tokens, H, D, dtype=torch.float16, generator=gen)
    g = F.logsigmoid(
        torch.randn(1, total_tokens, H, dtype=torch.float32, generator=gen)
    )
    beta = torch.rand(1, total_tokens, H, dtype=torch.float16, generator=gen)
    h0 = (0.05 * torch.randn(len(cu_list) - 1, H, D, D, generator=gen)).to(
        torch.float16
    )
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "h0": h0, "scale": D**-0.5}


def _run_and_check(inputs, cu_list, label, device=None, check_h_boundaries=False):
    """Run the wrapper once and verify o / final_state / h against native.

    ``h[c]`` stores the recurrent state before chunk ``c`` (i.e. after
    processing chunks ``0..c-1``), while ``final_state`` is the state after
    the last chunk of each sequence. The h-boundary check re-runs the native
    reference on every chunk prefix, so it is opt-in to keep runtime bounded.
    """
    device = device or torch.device("npu")
    total_tokens = cu_list[-1]

    def to_npu(t):
        return t.to(device)

    cu = torch.tensor(cu_list, dtype=torch.long, device=device)
    g_sum, o, a_inv, final_state, w, h, v_new = run_mega_chunk_gdn(
        q=to_npu(inputs["q"]),
        k=to_npu(inputs["k"]),
        v=to_npu(inputs["v"]),
        g=to_npu(inputs["g"]),
        beta=to_npu(inputs["beta"]),
        scale=inputs["scale"],
        initial_state=to_npu(inputs["h0"]),
        output_final_state=True,
        cu_seqlens=cu,
        return_internals=True,
    )
    torch.npu.synchronize()

    assert g_sum.shape[1] == total_tokens
    assert o.shape[1] == total_tokens
    assert a_inv.shape[1] == total_tokens
    assert w.shape[1] == total_tokens
    assert v_new.shape[1] == total_tokens
    assert h.dim() == 5 and h.shape[0] == 1

    expected_o, expected_states = _native_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        cu_list,
        initial_state=inputs["h0"],
        output_final_state=True,
    )
    _assert_close(f"{label}/o", o, expected_o)
    _assert_close(f"{label}/final_state", final_state, expected_states)

    # h[c] stores the recurrent state BEFORE chunk c (post-chunk c-1), so
    # h[chunk_offset + c] for c >= 1 must equal the final state of the native
    # reference run on the prefix ending at that chunk boundary.
    if not check_h_boundaries:
        return
    chunk_offset = 0
    for seq_idx, (start, end) in enumerate(zip(cu_list, cu_list[1:])):
        seg = end - start
        n_chunks = -(-seg // CHUNK_SIZE)
        for c in range(1, n_chunks):
            boundary = start + c * CHUNK_SIZE
            _, prefix_state = _native_reference(
                inputs["q"][:, :boundary],
                inputs["k"][:, :boundary],
                inputs["v"][:, :boundary],
                inputs["g"][:, :boundary],
                inputs["beta"][:, :boundary],
                cu_list[: seq_idx + 1] + [boundary],
                initial_state=inputs["h0"][: seq_idx + 1],
                output_final_state=True,
            )
            _assert_close(
                f"{label}/h_boundary_seq{seq_idx}_chunk{c}",
                h[0, chunk_offset + c],
                prefix_state[seq_idx],
            )
        chunk_offset += n_chunks


def test_slab_grow_and_shrink_keep_correctness():
    if not _op_registered():
        pytest.skip("mega_chunk_gdn op is not registered")

    H, Hg, D = 16, 4, 128
    # Grow the slab (256 -> 1024), then shrink (1024 -> 384): the slab must
    # grow exactly once and every later call must slice [:, :total_tokens].
    # The 1024-token shape also exercises the h-boundary check, which re-runs
    # the native reference on every chunk prefix (kept small on purpose).
    shapes = [(256, "grow1", False), (1024, "grow2", True), (384, "shrink", False)]
    for total_tokens, name, check_h in shapes:
        cu_list = [0, total_tokens // 4, total_tokens]
        inputs = _make_inputs(total_tokens, cu_list, H, Hg, D, seed=total_tokens)
        _run_and_check(
            inputs, cu_list, f"slab_{name}", check_h_boundaries=check_h
        )


def test_meta_cache_toggle_keeps_outputs_identical():
    if not _op_registered():
        pytest.skip("mega_chunk_gdn op is not registered")

    H, Hg, D = 16, 4, 128
    total_tokens = 1024
    cu_list = [0, 512, 1024]
    inputs = _make_inputs(total_tokens, cu_list, H, Hg, D, seed=7)
    device = torch.device("npu")
    cu = torch.tensor(cu_list, dtype=torch.long, device=device)

    kwargs = dict(
        q=inputs["q"].to(device),
        k=inputs["k"].to(device),
        v=inputs["v"].to(device),
        g=inputs["g"].to(device),
        beta=inputs["beta"].to(device),
        scale=inputs["scale"],
        initial_state=inputs["h0"].to(device),
        output_final_state=True,
        return_internals=True,
    )

    try:
        set_gdn_meta_cache(False)
        _, o_off, _, fs_off, _, _, _ = run_mega_chunk_gdn(cu_seqlens=cu, **kwargs)
        torch.npu.synchronize()

        set_gdn_meta_cache(True)
        _, o_on, _, fs_on, _, _, _ = run_mega_chunk_gdn(cu_seqlens=cu, **kwargs)
        torch.npu.synchronize()
    finally:
        set_gdn_meta_cache(True)

    assert torch.equal(o_off, o_on), "o differs with meta cache toggled"
    assert torch.equal(fs_off, fs_on), (
        "final_state differs with meta cache toggled"
    )


def test_repeated_identical_cu_seqlens_content():
    if not _op_registered():
        pytest.skip("mega_chunk_gdn op is not registered")

    # Same cu_seqlens content but a fresh device tensor each call: exercises
    # the content-keyed cu_seqlens D2H cache and the slab hit path.
    H, Hg, D = 16, 4, 128
    total_tokens = 1536
    cu_list = [0, 400, 900, 1536]
    inputs = _make_inputs(total_tokens, cu_list, H, Hg, D, seed=11)
    device = torch.device("npu")

    first = None
    for i in range(3):
        cu = torch.tensor(cu_list, dtype=torch.long, device=device)
        _, o, _, final_state, _, _, _ = run_mega_chunk_gdn(
            q=inputs["q"].to(device),
            k=inputs["k"].to(device),
            v=inputs["v"].to(device),
            g=inputs["g"].to(device),
            beta=inputs["beta"].to(device),
            scale=inputs["scale"],
            initial_state=inputs["h0"].to(device),
            output_final_state=True,
            cu_seqlens=cu,
            return_internals=True,
        )
        torch.npu.synchronize()
        if first is None:
            first = (o.clone(), final_state.clone())
        else:
            assert torch.equal(first[0], o), f"o drifted on call {i}"
            assert torch.equal(first[1], final_state), (
                f"final_state drifted on call {i}"
            )

    expected_o, _ = _native_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        cu_list,
        initial_state=inputs["h0"],
    )
    _assert_close("repeated_cu/o", first[0], expected_o)


def test_return_internals_false_contract():
    if not _op_registered():
        pytest.skip("mega_chunk_gdn op is not registered")

    H, Hg, D = 16, 4, 128
    total_tokens = 777  # not a multiple of CHUNK_SIZE: exercises the tail
    cu_list = [0, total_tokens]
    inputs = _make_inputs(total_tokens, cu_list, H, Hg, D, seed=3)
    device = torch.device("npu")
    cu = torch.tensor(cu_list, dtype=torch.long, device=device)

    g_sum, o, a_inv, final_state, w, h, v_new = run_mega_chunk_gdn(
        q=inputs["q"].to(device),
        k=inputs["k"].to(device),
        v=inputs["v"].to(device),
        g=inputs["g"].to(device),
        beta=inputs["beta"].to(device),
        scale=inputs["scale"],
        initial_state=inputs["h0"].to(device),
        output_final_state=True,
        cu_seqlens=cu,
        return_internals=False,
    )
    torch.npu.synchronize()

    # Contract of return_internals=False: A_inv / w / v_new are None (their
    # fp16->fp16 casts are skipped) while o / final_state / h stay usable.
    assert a_inv is None and w is None and v_new is None
    assert g_sum.dtype == torch.float32 and g_sum.shape[1] == total_tokens
    assert o.dtype == torch.float16 and o.shape[1] == total_tokens
    assert final_state.dtype == torch.float32
    assert h.dtype == torch.float16

    expected_o, _ = _native_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        cu_list,
        initial_state=inputs["h0"],
    )
    _assert_close("return_internals_false/o", o, expected_o)


def test_clear_workspace_cache_then_call():
    if not _op_registered():
        pytest.skip("mega_chunk_gdn op is not registered")

    H, Hg, D = 16, 4, 128
    total_tokens = 640
    cu_list = [0, 256, 640]

    for phase in ("before_clear", "after_clear"):
        inputs = _make_inputs(total_tokens, cu_list, H, Hg, D, seed=5)
        _run_and_check(inputs, cu_list, f"clear_{phase}")
        clear_mega_workspace_cache()
        torch.npu.empty_cache()


def test_host_overhead_not_regressed():
    """End-to-end gate: the cached wrapper must not be slower than upstream.

    Runs the SAME AscendC kernel under two wrapper variants (upstream-style
    per-call allocations vs. the cached wrapper) and compares end-to-end wall
    time, so kernel cost cancels out and any wrapper overhead delta shows.
    Gated behind an env flag because it is timing sensitive.
    """
    if os.environ.get("SGLANG_NPU_RUN_META_CACHE_PERF", "0") != "1":
        pytest.skip("set SGLANG_NPU_RUN_META_CACHE_PERF=1 to run the perf gate")
    if not _op_registered():
        pytest.skip("mega_chunk_gdn op is not registered")

    from sgl_kernel_npu.fla.mega_chunk_gdn import (
        _block_dim,
        _masks,
        _minus_identity,
    )

    H, Hg, D = 16, 4, 128
    total_tokens = 8192
    cu_list = [0, total_tokens // 2, total_tokens]
    device = torch.device("npu")
    num_value_heads = H
    num_seqs = len(cu_list) - 1
    num_chunks = sum(-(-(b - a) // CHUNK_SIZE) for a, b in zip(cu_list, cu_list[1:]))

    inputs = _make_inputs(total_tokens, cu_list, H, Hg, D, seed=1)
    q = inputs["q"].to(device)
    k = inputs["k"].to(device)
    v = inputs["v"].to(device)
    g = inputs["g"].to(device)
    beta = inputs["beta"].to(device)
    h0 = inputs["h0"].to(device)
    cu = torch.tensor(cu_list, dtype=torch.long, device=device)
    scale = inputs["scale"]

    device_type, device_index = device.type, 0 if device.index is None else device.index
    mask_lower, mask_full = _masks(device_type, device_index)
    minus_identity = _minus_identity(device_type, device_index)
    block_dim = _block_dim(device)

    def upstream_call():
        """Verbatim upstream per-call allocation pattern + same kernel launch."""
        cu32 = cu.to(torch.int32)
        _ = cu.cpu().tolist()  # D2H sync every call (upstream behavior)
        num_matrices = num_chunks * num_value_heads

        g_sum = torch.empty_like(g, dtype=torch.float32)
        g_t = torch.empty(num_value_heads, total_tokens, device=device, dtype=torch.float32)
        beta_t = torch.empty(num_value_heads, total_tokens, device=device, dtype=torch.float16)
        A = torch.zeros(1, total_tokens, num_value_heads, CHUNK_SIZE, device=device, dtype=torch.float16)
        A_inv_f32 = torch.zeros(1, total_tokens, num_value_heads, CHUNK_SIZE, device=device, dtype=torch.float32)
        A_inv = torch.zeros_like(A)
        w = torch.empty_like(v)
        u = torch.empty_like(v)
        h = torch.zeros(num_chunks * num_value_heads, D, D, device=device, dtype=torch.float16)
        v_new = torch.empty_like(v)
        final_state = torch.zeros(num_seqs * num_value_heads, D, D, device=device, dtype=torch.float16)
        out = torch.empty_like(v)

        kkt = torch.zeros(block_dim * 2, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
        wy_a1 = torch.zeros(block_dim, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
        wy_a2 = torch.zeros_like(wy_a1)
        h_ws = torch.zeros(block_dim * 4, D, D, device=device, dtype=torch.float16)
        o_qk = torch.zeros(block_dim, CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
        o_qs = torch.zeros(block_dim, CHUNK_SIZE, D, device=device, dtype=torch.float16)
        o_gated = torch.zeros_like(o_qk)

        torch.ops.npu.mega_chunk_gdn(
            q, k, v, g, beta, mask_lower, mask_full, minus_identity, cu32,
            out, g_sum, g_t, beta_t, A, A_inv_f32, A_inv, w, u, h, v_new,
            final_state, h0, True,
            kkt, wy_a1, wy_a2, h_ws, o_qk, o_qs, o_gated,
            block_dim, num_seqs, total_tokens, total_tokens, num_matrices,
        )
        return out

    def cached_call():
        _, o, _, _, _, _, _ = run_mega_chunk_gdn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            return_internals=False,
        )
        return o

    def wall_time(fn, iters=5):
        fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - start) / iters * 1e3

    upstream_ms = wall_time(upstream_call)
    cached_ms = wall_time(cached_call)

    # Sanity: the upstream replica must be wired identically to the wrapper.
    _, o_cached, _, _, _, _, _ = run_mega_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        return_internals=False,
    )
    out_upstream = upstream_call()
    torch.npu.synchronize()
    assert torch.allclose(out_upstream * scale, o_cached, atol=1e-2), (
        "upstream replica diverged from the cached wrapper in the perf gate"
    )

    print(f"\nwall/call upstream={upstream_ms:.3f} ms cached={cached_ms:.3f} ms")
    # End-to-end the cached wrapper must not regress beyond run-to-run noise.
    assert cached_ms < upstream_ms * 1.10, (
        f"cached wrapper regressed end-to-end: {cached_ms:.3f} ms "
        f">= 1.10x upstream {upstream_ms:.3f} ms"
    )


def test_env_toggle_consistent_with_runtime_toggle():
    """set_gdn_meta_cache must agree with the module-level enabled flag."""
    previous = is_gdn_meta_cache_enabled()
    try:
        set_gdn_meta_cache(False)
        assert not is_gdn_meta_cache_enabled()
        set_gdn_meta_cache(True)
        assert is_gdn_meta_cache_enabled()
    finally:
        set_gdn_meta_cache(previous)
