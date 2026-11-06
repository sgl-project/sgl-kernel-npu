"""Triton implementation of the GDN gate preprocessing.

The kernel body is a byte-for-byte copy of the legacy
``sgl_kernel_npu/fla/fused_gdn_gating.py::fused_gdn_gating_kernel``; only the
``get_device_properties`` import and the wrapper assertions are new.
"""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties

# Usable unified buffer budget in bytes: reserve half of the
# 192 KiB capacity for double buffering, leaving 96 KiB per buffer.
UNIFIED_BUFFER_SIZE = 96 * 1024


@triton.jit(
    do_not_specialize=[
        "seq_len",
        "NUM_HEADS",
        "NUM_BATCHES",
        "beta",
        "threshold",
        "ROW_ITER",
    ]
)
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS,
    NUM_BATCHES,
    beta,
    threshold,
    BLK_HEADS: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    ROW_ITER,
):
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    COL_ITER = tl.cdiv(NUM_HEADS, BLK_HEADS)

    for row_idx in range(0, ROW_ITER):
        batch_off = (
            i_b * ROW_ITER * BLK_BATCHES
            + row_idx * BLK_BATCHES
            + tl.arange(0, BLK_BATCHES)
        )

        for col_idx in range(0, COL_ITER):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)

            off = (
                batch_off[:, None] * seq_len * NUM_HEADS
                + i_s * NUM_HEADS
                + head_off[None, :]
            )
            head_mask = head_off < NUM_HEADS
            mask = head_mask[None, :] & (batch_off[:, None] < NUM_BATCHES)

            blk_A_log = tl.load(A_log + head_off, mask=head_mask)
            blk_a = tl.load(a + off, mask=mask)
            blk_b = tl.load(b + off, mask=mask)
            blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
            softplus_x = tl.where(
                beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
            )

            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)

            # compute beta_output = sigmoid(b)
            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
            tl.store(
                beta_output + off,
                blk_beta_output.to(beta_output.dtype.element_ty),
                mask=mask,
            )


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Return (g, sigmoid(b)) for [rows, num_heads] gating inputs."""
    batch, num_heads = a.shape
    seq_len = 1

    # New: reject inputs the contiguous-layout kernel cannot represent.
    assert (
        b.shape == a.shape
    ), f"b shape {tuple(b.shape)} must match a shape {tuple(a.shape)}"
    assert (
        a.is_contiguous() and b.is_contiguous()
    ), f"a and b must be contiguous; got strides {a.stride()} and {b.stride()}"
    assert (
        A_log.numel() == num_heads
    ), f"A_log must have {num_heads} elements, got {A_log.numel()}"
    assert (
        dt_bias.numel() == num_heads
    ), f"dt_bias must have {num_heads} elements, got {dt_bias.numel()}"

    num_cores = get_device_properties()[1]

    BLK_HEADS = 8
    COL_ITER = triton.cdiv(num_heads, BLK_HEADS)

    elem_size = a.element_size()
    max_ub_batches = int((UNIFIED_BUFFER_SIZE * 0.95) / (BLK_HEADS * elem_size))
    if batch <= num_cores:
        progs = batch
        BLK_BATCHES = 1
        ROW_ITER = 1
    else:
        progs = num_cores
        # Heads are processed in BLK_HEADS-wide tiles, so the UB estimate
        # depends on one tile rather than the total number of heads.
        FACTOR = 8
        calc_blk_batches = (
            triton.next_power_of_2(
                triton.cdiv(
                    int(UNIFIED_BUFFER_SIZE * 0.95), FACTOR * BLK_HEADS * elem_size
                )
            )
            // 2
        )
        BLK_BATCHES = max(1, min(calc_blk_batches, max_ub_batches, 64))
        row_per_core = triton.cdiv(batch, progs)
        ROW_ITER = triton.cdiv(row_per_core, BLK_BATCHES)

    g = torch.empty(batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(batch, num_heads, dtype=b.dtype, device=b.device)

    grid = (progs, seq_len)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        batch,
        beta,
        threshold,
        BLK_HEADS=BLK_HEADS,
        BLK_BATCHES=BLK_BATCHES,
        ROW_ITER=ROW_ITER,
    )
    return g, beta_output
