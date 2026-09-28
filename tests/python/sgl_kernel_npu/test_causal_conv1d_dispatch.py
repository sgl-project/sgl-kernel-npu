"""Dispatch and graph-capture tests for the AscendC causal-conv1d fast path.

The v2 wrappers (causal_conv1d_fn_v2 / causal_conv1d_update_npu_v2) prefer
the ``torch.ops.npu.causal_conv1d`` AscendC op and silently fall back to the
torch/Triton composition when the op is unavailable or a precondition is
unmet. These tests pin that behavior:

- the op path is actually taken (recording probe around the op handle),
  with run_mode=1 on the decode step;
- the SGL_NPU_DISABLE_ASCENDC_CONV1D kill switch forces the fallback and
  re-enables cleanly;
- the forced fallback matches the op path's outputs and state writes
  (pad rows excluded -- their outputs are unspecified);
- both op run modes capture into an NPU graph and replay correctly on
  mutated inputs (sglang's decode runs under graph replay).

Skipped entirely when the AscendC op is not registered in this build.
All references run on CPU in fp32 (fp32 F.conv1d on NPU can produce NaNs).
All op calls use fp16, matching the sibling test_causal_conv1d_v2.py: the
op's host-side tiling cache is keyed by shape but (in currently deployed
builds) not by dtype, so mixing dtypes over identical shapes in one
process would poison later calls. bf16 graph capture is exercised by the
sglang e2e service itself.
"""

import os

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.npu available

import sgl_kernel_npu.mamba.causal_conv1d as cc
from sgl_kernel_npu.mamba.causal_conv1d import (
    PAD_SLOT_ID,
    causal_conv1d_fn_v2,
    causal_conv1d_update_npu_v2,
)

device = "npu"
torch.manual_seed(11)

DIM, WIDTH = 128, 4
STATE_LEN = WIDTH - 1
PAD = PAD_SLOT_ID

op = cc._ascendc_conv1d_op()
pytestmark = pytest.mark.skipif(
    op is None, reason="AscendC causal_conv1d op not registered in this build"
)


@pytest.fixture()
def restore_op_handle():
    """Save the resolved op handle and the kill-switch env; restore after."""
    saved = cc._ASCENDC_CONV1D_OP
    saved_env = os.environ.get("SGL_NPU_DISABLE_ASCENDC_CONV1D")
    yield
    if saved_env is None:
        os.environ.pop("SGL_NPU_DISABLE_ASCENDC_CONV1D", None)
    else:
        os.environ["SGL_NPU_DISABLE_ASCENDC_CONV1D"] = saved_env
    cc._ASCENDC_CONV1D_OP = saved


def _record_calls():
    """Replace the cached op handle with a recording wrapper."""
    real_op = cc._ascendc_conv1d_op()
    calls = {"n": 0, "run_modes": []}

    def recording_op(*args, **kwargs):
        calls["n"] += 1
        calls["run_modes"].append(kwargs.get("run_mode", 0))
        return real_op(*args, **kwargs)

    cc._ASCENDC_CONV1D_OP = recording_op
    return calls


def test_dispatch_prefill_takes_op(restore_op_handle):
    lens, starts = [5, 2, 1], [0, 5, 7, 8]
    cu = starts[-1]
    x = (torch.randn(DIM, cu) * 0.3).to(torch.float16).to(device).contiguous()
    pool = (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    w = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    qsl = torch.tensor(starts, device=device, dtype=torch.int32)
    cidx = torch.tensor([3, 5, 6], device=device, dtype=torch.int32)

    calls = _record_calls()
    out = causal_conv1d_fn_v2(
        x, w, bias, qsl, cidx, None, pool, None, pad_slot_id=PAD
    )
    torch.npu.synchronize()
    assert calls["n"] == 1
    assert calls["run_modes"] == [0]  # prefill run mode

    # has_initial_state=None -> all-fresh: zero-init CPU reference.
    for i, length in enumerate(lens):
        toks = x[:, starts[i] : starts[i + 1]].cpu().float()
        virt = torch.cat([torch.zeros(DIM, STATE_LEN), toks], dim=1)
        ref = F.conv1d(
            virt.unsqueeze(0),
            w.cpu().float().unsqueeze(1),
            bias.cpu().float(),
            groups=DIM,
        )[0]
        got = out[:, starts[i] : starts[i + 1]].cpu().float()
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
        got_state = pool[cidx[i].item()].cpu().float()
        torch.testing.assert_close(
            got_state, virt[:, -STATE_LEN:].t(), atol=2e-2, rtol=2e-2
        )


def test_dispatch_decode_takes_op(restore_op_handle):
    x = (torch.randn(4, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    pool = (
        (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    )
    w = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    idx = torch.tensor([1, 3, PAD, 6], device=device, dtype=torch.int32)

    calls = _record_calls()
    causal_conv1d_update_npu_v2(
        x, pool, w, bias, None, conv_state_indices=idx, pad_slot_id=PAD
    )
    torch.npu.synchronize()
    assert calls["n"] == 1
    assert calls["run_modes"] == [1]  # decode run mode


def test_env_kill_switch(restore_op_handle):
    os.environ["SGL_NPU_DISABLE_ASCENDC_CONV1D"] = "1"
    cc._ASCENDC_CONV1D_OP = None  # force re-resolution
    assert cc._ascendc_conv1d_op() is None

    del os.environ["SGL_NPU_DISABLE_ASCENDC_CONV1D"]
    cc._ASCENDC_CONV1D_OP = None
    assert cc._ascendc_conv1d_op() is not None


def test_fallback_equivalence_decode(restore_op_handle):
    x = (torch.randn(4, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    pool = (
        (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    )
    w = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    idx = torch.tensor([1, 3, PAD, 6], device=device, dtype=torch.int32)

    # Both paths must start from the same initial pool state; clone it
    # BEFORE the fallback runs (it writes new states in place).
    pool_op = pool.clone()
    # Forced fallback: the Triton kernel overwrites x in place, so feed it a
    # clone.
    cc._ASCENDC_CONV1D_OP = False
    out_fb = causal_conv1d_update_npu_v2(
        x.clone(), pool, w, bias, None, conv_state_indices=idx, pad_slot_id=PAD
    )
    torch.npu.synchronize()

    # Op path on identical inputs.
    cc._ASCENDC_CONV1D_OP = op
    out_op = causal_conv1d_update_npu_v2(
        x.clone(), pool_op, w, bias, None, conv_state_indices=idx, pad_slot_id=PAD
    )
    torch.npu.synchronize()

    # Pad-row outputs are unspecified on both paths; compare valid rows.
    valid_rows = [0, 1, 3]
    torch.testing.assert_close(
        out_fb[valid_rows].float(), out_op[valid_rows].float(),
        atol=4e-2, rtol=4e-2,
    )
    torch.testing.assert_close(
        pool.float(), pool_op.float(), atol=4e-2, rtol=4e-2
    )


def test_fallback_equivalence_prefill(restore_op_handle):
    lens, starts = [5, 2, 1], [0, 5, 7, 8]
    cu = starts[-1]
    x = (torch.randn(DIM, cu) * 0.3).to(torch.float16).to(device).contiguous()
    pool1 = (
        (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    )
    w = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    qsl = torch.tensor(starts, device=device, dtype=torch.int32)
    cidx = torch.tensor([3, 5, 6], device=device, dtype=torch.int32)
    has_init = torch.tensor([True, False, True], device=device, dtype=torch.bool)
    pool2 = pool1.clone()

    cc._ASCENDC_CONV1D_OP = False
    out_fb = causal_conv1d_fn_v2(
        x, w, bias, qsl, cidx, has_init, pool1, "silu", pad_slot_id=PAD
    )
    cc._ASCENDC_CONV1D_OP = op
    out_op = causal_conv1d_fn_v2(
        x, w, bias, qsl, cidx, has_init, pool2, "silu", pad_slot_id=PAD
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        out_fb.float(), out_op.float(), atol=4e-2, rtol=4e-2
    )
    torch.testing.assert_close(
        pool1.float(), pool2.float(), atol=4e-2, rtol=4e-2
    )


def test_graph_capture_decode(restore_op_handle):
    b = 4
    x = (torch.randn(b, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    pool = (
        (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    )
    w_wd = (torch.randn(WIDTH, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    idx = torch.tensor([1, 3, PAD, 6], device=device, dtype=torch.int32)

    def call():
        return op(
            x.unsqueeze(1), w_wd, pool, bias=bias, cache_indices=idx,
            activation_mode=0, pad_slot_id=PAD, run_mode=1,
        )

    # Warmup: exercises the op's internal tiling-capture registration.
    for _ in range(3):
        call()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y = call()

    # Mutate inputs in place (replay works on the same storage), replay.
    x2 = (torch.randn(b, DIM) * 0.3).to(torch.float16)
    pool_init = (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16)
    w2 = (torch.randn(WIDTH, DIM) * 0.3).to(torch.float16)
    bias2 = (torch.randn(DIM) * 0.3).to(torch.float16)
    x.copy_(x2)
    pool.copy_(pool_init)
    w_wd.copy_(w2)
    bias.copy_(bias2)

    graph.replay()
    torch.npu.synchronize()
    for row, slot in [(0, 1), (1, 3), (3, 6)]:
        state = pool_init[slot].cpu().float().t()
        win = torch.cat([state, x2[row].cpu().float().unsqueeze(1)], dim=1)
        ref = F.conv1d(
            win.unsqueeze(0),
            w2.t().cpu().float().unsqueeze(1),
            bias2.cpu().float(),
            groups=DIM,
        )[0][:, -1]
        torch.testing.assert_close(
            y[row, 0].cpu().float(), ref, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            pool[slot].cpu().float(), win[:, -STATE_LEN:].t(),
            atol=2e-2, rtol=2e-2,
        )

    # Second replay after another input swap: the rolling window advanced
    # with replay 1, so the reference builds on replay 1's final state.
    x3 = (torch.randn(b, DIM) * 0.3).to(torch.float16)
    x.copy_(x3)
    graph.replay()
    torch.npu.synchronize()
    pre_state = torch.cat(
        [pool_init[1].cpu().float().t()[:, 1:], x2[0].cpu().float().unsqueeze(1)],
        dim=1,
    )
    win2 = torch.cat([pre_state, x3[0].cpu().float().unsqueeze(1)], dim=1)
    ref2 = F.conv1d(
        win2.unsqueeze(0),
        w2.t().cpu().float().unsqueeze(1),
        bias2.cpu().float(),
        groups=DIM,
    )[0][:, -1]
    torch.testing.assert_close(
        y[0, 0].cpu().float(), ref2, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        pool[1].cpu().float(), win2[:, -STATE_LEN:].t(), atol=2e-2, rtol=2e-2
    )


def test_graph_capture_prefill(restore_op_handle):
    lens, starts = [5, 3], [0, 5, 8]
    cu = starts[-1]
    xp = (torch.randn(cu, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    pool = (
        (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    )
    w = (torch.randn(WIDTH, DIM) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    qsl = torch.tensor(starts, device=device, dtype=torch.int32)
    cidx = torch.tensor([2, 5], device=device, dtype=torch.int32)
    has_init = torch.tensor([True, False], device=device, dtype=torch.bool)

    def call():
        return op(
            xp, w, pool, bias=bias, query_start_loc=qsl, cache_indices=cidx,
            has_initial_state=has_init, activation_mode=0, pad_slot_id=PAD,
        )

    for _ in range(3):
        call()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = call()

    xp2 = (torch.randn(cu, DIM) * 0.3).to(torch.float16)
    pool_init = (torch.randn(8, STATE_LEN, DIM) * 0.3).to(torch.float16)
    xp.copy_(xp2)
    pool.copy_(pool_init)
    graph.replay()
    torch.npu.synchronize()

    for i, length in enumerate(lens):
        slot = cidx[i].item()
        init = (
            pool_init[slot].cpu().float().t()
            if has_init[i]
            else torch.zeros(DIM, STATE_LEN)
        )
        toks = xp2[starts[i] : starts[i + 1]].cpu().float().t()
        virt = torch.cat([init, toks], dim=1)
        ref = F.conv1d(
            virt.unsqueeze(0),
            w.t().cpu().float().unsqueeze(1),
            bias.cpu().float(),
            groups=DIM,
        )[0]
        torch.testing.assert_close(
            out[starts[i] : starts[i + 1]].cpu().float().t(), ref,
            atol=2e-2, rtol=2e-2,
        )
        torch.testing.assert_close(
            pool[slot].cpu().float(), virt[:, -STATE_LEN:].t(),
            atol=2e-2, rtol=2e-2,
        )
