from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit(do_not_specialize=["batch", "seq_len"])
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    batch,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    core, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    mask = head_off < NUM_HEADS

    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)

    for i_b in tl.range(core, batch, tl.num_programs(0)):
        off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off

        blk_a = tl.load(a + off, mask=mask)
        blk_b = tl.load(b + off, mask=mask)

        x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )

        blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
        blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))

        tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
        tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating_npu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    batch, num_heads = a.shape
    seq_len = 1

    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)

    device = torch.npu.current_device()
    num_cores = driver.active.utils.get_device_properties(device)["num_vectorcore"]

    grid = (
        triton.cdiv(num_cores, triton.cdiv(num_heads, 8)),
        seq_len,
        triton.cdiv(num_heads, 8),
    )

    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        batch,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        multibuffer=True,
        num_warps=1,
    )
    return g, beta_output
