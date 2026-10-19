"""Unit tests for the Python-layer causal-conv1d ops on NPU.

Covers the paths NOT exercised by test_conv1d_prefill.py /
test_conv1d_update.py (which target the ``torch.ops.npu.causal_conv1d``
CANN op):

- causal_conv1d_fn_v2 / causal_conv1d_update_npu_v2: the window-major
  pool ops used by sglang's short-conv hybrid models (LFM2 / LFM2-MoE)
- causal_conv1d_fn_npu: the legacy varlen prefill path (GDN / Mamba2
  families), regression tests for the pad-slot and final-state-offset
  fixes

All references run on CPU in fp32: fp32 F.conv1d on this NPU has been
observed to produce NaNs, so the host is the source of truth.
"""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.npu available

from sgl_kernel_npu.mamba.causal_conv1d import (
    PAD_SLOT_ID,
    causal_conv1d_fn_npu,
    causal_conv1d_fn_v2,
    causal_conv1d_update_npu_v2,
)

device = "npu"
torch.manual_seed(42)

DIM, WIDTH = 128, 4
STATE_LEN = WIDTH - 1
PAD = PAD_SLOT_ID

UPDATE_CASES = [
    # (tag, pool_shape, with_indices, use_bias, silu, seqlen)
    ("wm_bias_s1", "s_cd", True, True, False, 1),
    ("wm_bias_silu_s3", "s_cd", True, True, True, 3),
    ("wm_nobias_s1", "s_cd", True, False, False, 1),
    ("cm_bias_s1", "cd_s", True, True, False, 1),
    ("cm_bias_silu_s3", "cd_s", True, True, True, 3),
    ("fb_bias_s1", "cd_s", False, True, False, 1),
    ("fb_bias_silu_s3", "s_cd", False, True, True, 3),
    ("fb_nobias_s3", "s_cd", False, False, False, 3),
]

PREFILL_CASES = [
    # (tag, lens, has_init, pad_rows, silu, use_bias)
    ("mix", [5, 2, 1, 3], [True, False, True, False],
     [False, False, True, False], False, True),
    ("mix_silu", [7, 1, 3], [False, True, False],
     [False, False, False], True, False),
    ("trailing_zero_len", [4, 3, 0], [True, False, False],
     [False, False, False], False, True),
]


def _ref_update(x, states, w, bias, silu):
    """x: (b, s, dim); states: (b, state_len, dim); w: (dim, width).

    Returns (out, new_states) computed on CPU in fp32.
    """
    x = x.cpu().float()
    states = states.cpu().float()
    w = w.cpu().float()
    b, s, dim = x.shape
    sl = w.shape[1] - 1
    outs, new_states = [], []
    for i in range(b):
        win = torch.cat([states[i].t(), x[i].t()], dim=1)  # (dim, sl+s)
        out = F.conv1d(
            win.unsqueeze(0), w.unsqueeze(1),
            None if bias is None else bias.cpu().float(), groups=dim,
        ).squeeze(0)
        if silu:
            out = F.silu(out)
        outs.append(out.t())
        new_states.append(win[:, -sl:].t())
    return torch.stack(outs), torch.stack(new_states)


def _ref_prefill(x, starts, pool_rows, has_init, w, bias, silu):
    """x: (dim, cu); pool_rows: (b, state_len, dim). -> (outs, finals)."""
    x = x.cpu().float()
    pool_rows = pool_rows.cpu().float()
    w = w.cpu().float()
    dim = x.shape[0]
    sl = w.shape[1] - 1
    outs, finals = [], []
    for i in range(len(starts) - 1):
        length = starts[i + 1] - starts[i]
        toks = x[:, starts[i]:starts[i + 1]]
        init = pool_rows[i].t() if has_init[i] else torch.zeros(dim, sl)
        if length == 0:
            outs.append(None)
            finals.append(init)
            continue
        virtual = torch.cat([init, toks], dim=1)
        out = F.conv1d(
            virtual.unsqueeze(0), w.unsqueeze(1),
            None if bias is None else bias.cpu().float(), groups=dim,
        ).squeeze(0)
        if silu:
            out = F.silu(out)
        outs.append(out)
        finals.append(virtual[:, -sl:])
    return outs, finals


@pytest.mark.parametrize(
    "tag,pool_shape,with_indices,use_bias,silu,seqlen", UPDATE_CASES
)
def test_update_npu_v2(tag, pool_shape, with_indices, use_bias, silu, seqlen):
    batch, n_real_slots = 4, 8
    n_slots = batch if not with_indices else n_real_slots
    x2d = seqlen == 1
    x = torch.randn(batch, dim := DIM) if x2d else torch.randn(batch, DIM, seqlen)
    x = (x * 0.3).to(torch.float16).to(device).contiguous()
    x0 = x.clone()
    pool = (
        torch.randn(n_slots, DIM, STATE_LEN) if pool_shape == "cd_s"
        else torch.randn(n_slots, STATE_LEN, DIM)
    )
    pool = (pool * 0.3).to(torch.float16).to(device).contiguous()
    pool0 = pool.clone()
    weight = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (
        (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
        if use_bias else None
    )
    idx = torch.tensor([1, 3, PAD, 6], device=device, dtype=torch.int32)

    out = causal_conv1d_update_npu_v2(
        x, pool, weight, bias, "silu" if silu else None,
        conv_state_indices=idx if with_indices else None, pad_slot_id=PAD,
    )
    torch.npu.synchronize()

    if with_indices:
        sel, valid = [1, 3, 6], [0, 1, 3]
    else:
        sel, valid = list(range(batch)), list(range(batch))
    st0 = pool0[sel]
    st0 = st0.transpose(1, 2) if pool_shape == "cd_s" else st0
    xs = x0[valid]
    xs3 = (xs.unsqueeze(1) if x2d else xs.transpose(1, 2)).contiguous().cpu()
    ref_out, ref_state = _ref_update(xs3, st0.contiguous().cpu(), weight.cpu(), bias, silu)
    ref_out = ref_out.to(device)
    ref_cmp = ref_out.squeeze(1) if x2d else ref_out.transpose(1, 2)

    torch.testing.assert_close(
        out[valid].float(), ref_cmp.float(), atol=2e-2, rtol=2e-2
    )
    got_state = pool[sel]
    got_state = got_state.transpose(1, 2) if pool_shape == "cd_s" else got_state
    torch.testing.assert_close(
        got_state.float(), ref_state.to(device).float(), atol=2e-2, rtol=2e-2
    )
    if with_indices:
        # pad row must not touch any real slot (slot 0 is the reserved dummy)
        untouched = [k for k in range(n_real_slots) if k not in (0, 1, 3, 6)]
        torch.testing.assert_close(
            pool[untouched].float(), pool0[untouched].float(), atol=0.0, rtol=0.0
        )


@pytest.mark.parametrize(
    "tag,lens,has_init,pad_rows,silu,use_bias", PREFILL_CASES
)
def test_fn_v2(tag, lens, has_init, pad_rows, silu, use_bias):
    n_slots = 10
    starts = [0]
    for length in lens:
        starts.append(starts[-1] + length)
    cu = starts[-1]
    x = (torch.randn(DIM, cu) * 0.3).to(torch.float16).to(device).contiguous()
    x0 = x.clone()
    pool = (
        torch.randn(n_slots, STATE_LEN, DIM) * 0.3
    ).to(torch.float16).to(device).contiguous()
    pool0 = pool.clone()
    weight = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (
        (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
        if use_bias else None
    )
    qsl = torch.tensor(starts, device=device, dtype=torch.int32)
    slots = [2, 4, 6, 8][: len(lens)]
    cache_idx = torch.tensor(
        [PAD if pad_rows[i] else slots[i] for i in range(len(lens))],
        device=device, dtype=torch.int32,
    )
    hi = torch.tensor(has_init, device=device, dtype=torch.bool)

    out = causal_conv1d_fn_v2(
        x, weight, bias, qsl, cache_idx, hi, pool,
        "silu" if silu else None, pad_slot_id=PAD,
    )
    torch.npu.synchronize()
    assert out.shape == (DIM, cu)

    ref_outs, ref_finals = _ref_prefill(
        x0.cpu(), starts, pool0[slots].cpu(), has_init, weight.cpu(), bias, silu
    )
    for i in range(len(lens)):
        if pad_rows[i]:
            continue  # pad outputs are unspecified
        if lens[i] > 0:
            got = out[:, starts[i]:starts[i + 1]]
            torch.testing.assert_close(
                got.float(), ref_outs[i].to(device).float(), atol=2e-2, rtol=2e-2
            )
        torch.testing.assert_close(
            pool[slots[i]].float(),
            ref_finals[i].to(device).t().float(),
            atol=2e-2, rtol=2e-2,
        )
    untouched = [k for k in range(n_slots) if k != 0 and k not in slots]
    torch.testing.assert_close(
        pool[untouched].float(), pool0[untouched].float(), atol=0.0, rtol=0.0
    )


def test_fn_native_final_state_offset():
    """Regression for the mixed-batch final-state gather offset.

    When initial_states extend x by (width-1) tokens, the last (width-1)
    inputs of every sequence sit at [seqlens, seqlens+width-2) regardless
    of has_initial_state; the old formula shifted no-init rows back into
    the initial-state region.
    """
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_native

    lens = [5, 5, 2, 1]  # row3: seqlen < width-1 edge case
    has_init = [True, False, True, False]
    batch, max_t = len(lens), max(lens)

    x_pad = torch.zeros(batch, DIM, max_t)
    for i, length in enumerate(lens):
        x_pad[i, :, :length] = torch.randn(DIM, length)
    init_pool = torch.randn(batch, DIM, STATE_LEN)
    init_in = init_pool.clone()
    for i in range(batch):
        if not has_init[i]:
            init_in[i] = 0.0
    seqlens = torch.tensor(lens, dtype=torch.int64)
    weight = torch.randn(DIM, WIDTH) * 0.3
    bias = torch.randn(DIM) * 0.3
    hi = torch.tensor(has_init, dtype=torch.bool)

    # Extended path (mixed batch -> the bug scenario)
    out, finals = causal_conv1d_fn_native(
        x_pad.clone(), weight, bias, seqlens=seqlens, has_initial_state=hi,
        initial_states=init_in, return_final_states=True, activation=None,
    )
    for i in range(batch):
        length = lens[i]
        init = init_pool[i] if has_init[i] else torch.zeros(DIM, STATE_LEN)
        virt = torch.cat([init, x_pad[i, :, :length]], dim=1)
        ref = F.conv1d(
            virt.unsqueeze(0), weight.unsqueeze(1), bias, groups=DIM
        )[0]
        torch.testing.assert_close(out[i, :, :length], ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(finals[i], virt[:, -STATE_LEN:], atol=1e-4, rtol=1e-4)

    # Unextended path (all has_init False)
    out2, finals2 = causal_conv1d_fn_native(
        x_pad.clone(), weight, bias, seqlens=seqlens,
        has_initial_state=torch.zeros(batch, dtype=torch.bool),
        initial_states=None, return_final_states=True, activation=None,
    )
    for i in range(batch):
        length = lens[i]
        virt = torch.cat([torch.zeros(DIM, STATE_LEN), x_pad[i, :, :length]], dim=1)
        ref = F.conv1d(
            virt.unsqueeze(0), weight.unsqueeze(1), bias, groups=DIM
        )[0]
        torch.testing.assert_close(out2[i, :, :length], ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(finals2[i], virt[:, -STATE_LEN:], atol=1e-4, rtol=1e-4)


def test_fn_npu_pad_slots():
    """Regression for the legacy varlen prefill path: pad slots must not
    wrap around via negative indexing (read side and write side)."""
    lens = [5, 2, 3]
    has_init = [True, False, True]
    pad_rows = [False, True, False]  # row1 is a pad request
    starts = [0]
    for length in lens:
        starts.append(starts[-1] + length)
    cu = starts[-1]
    batch = len(lens)

    x = (torch.randn(DIM, cu) * 0.3).to(torch.float16).to(device).contiguous()
    x0 = x.clone()
    # legacy pool convention: (slots, dim, state_len), channel-major
    pool = (
        torch.randn(8, DIM, STATE_LEN) * 0.3
    ).to(torch.float16).to(device).contiguous()
    pool0 = pool.clone()
    weight = (torch.randn(DIM, WIDTH) * 0.3).to(torch.float16).to(device).contiguous()
    bias = (torch.randn(DIM) * 0.3).to(torch.float16).to(device).contiguous()
    qsl = torch.tensor(starts, device=device, dtype=torch.int32)
    slots = [2, 5, 7]
    cache_idx = torch.tensor(
        [PAD if pad_rows[i] else slots[i] for i in range(batch)],
        device=device, dtype=torch.int32,
    )
    hi = torch.tensor(has_init, device=device, dtype=torch.bool)

    out = causal_conv1d_fn_npu(
        x, weight, bias,
        query_start_loc=qsl,
        cache_indices=cache_idx,
        has_initial_state=hi,
        conv_states=pool,
        activation=None,
    )
    torch.npu.synchronize()
    assert out.shape == (DIM, cu)

    ref_outs, ref_finals = _ref_prefill(
        x0.cpu(), starts, pool0[slots].transpose(1, 2).cpu(),
        has_init, weight.cpu(), bias, silu=False,
    )
    for i in range(batch):
        if pad_rows[i]:
            continue
        torch.testing.assert_close(
            out[:, starts[i]:starts[i + 1]].float(),
            ref_outs[i].to(device).float(), atol=2e-2, rtol=2e-2,
        )
        # legacy pool is channel-major: state rows are (dim, state_len)
        torch.testing.assert_close(
            pool[slots[i]].float(),
            ref_finals[i].to(device).float(), atol=2e-2, rtol=2e-2,
        )
    # The last pool slot (index -1 in Python) is where a pad write would
    # land via negative indexing; it must stay untouched, as must every
    # other unclaimed slot.
    untouched = [k for k in range(8) if k not in slots]
    torch.testing.assert_close(
        pool[untouched].float(), pool0[untouched].float(), atol=0.0, rtol=0.0
    )
