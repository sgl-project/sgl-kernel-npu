import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def fused_gdn_gating_kernel_npu(
    g,
    A_log,
    a,
    dt_bias,
    batch,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    batch_off = i_b * BLK_BATCHES + tl.arange(0, BLK_BATCHES)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    head_mask = head_off < NUM_HEADS

    a_off = (
        batch_off[:, None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[None, :]
    )
    a_mask = (batch_off[:, None] < batch) & head_mask[None, :]

    blk_A_log = tl.load(A_log + head_off, mask=head_mask)
    blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

    blk_a = tl.load(a + a_off, mask=a_mask)

    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + a_off, blk_g.to(g.dtype.element_ty), mask=a_mask)


def fused_gdn_gating_npu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    seq_len = 1
    g = torch.empty_like(a, dtype=torch.float32)

    _, num_vectorcore = get_device_properties()
    NUM_BLK_BATCHES = triton.cdiv(num_vectorcore, triton.cdiv(num_heads, 8))
    BLK_BATCHES = triton.cdiv(batch, NUM_BLK_BATCHES)
    grid = (NUM_BLK_BATCHES, seq_len, triton.cdiv(num_heads, 8))

    fused_gdn_gating_kernel_npu[grid](
        g,
        A_log,
        a,
        dt_bias,
        batch,
        seq_len,
        num_heads,
        beta,
        threshold,
        BLK_BATCHES=BLK_BATCHES,
        BLK_HEADS=8,
        num_warps=1,
    )
    return g
