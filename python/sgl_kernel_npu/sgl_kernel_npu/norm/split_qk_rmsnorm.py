import torch
import triton
import triton.language as tl

from functools import cache
from typing import Any, Dict, Tuple


@cache
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = (
        triton.runtime.driver.active.utils.get_device_properties(device)
    )

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


# before
@triton.jit
def split_qk_rmsnorm_kernel(
        input_ptr,
        q_ptr,
        k_nope_ptr,
        k_pe_ptr,
        q_weight_ptr,
        k_weight_ptr,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        total_hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    row_start = row_pid * total_hidden_size
    offsets = tl.arange(0, BLOCK_SIZE)

    # q
    q = tl.load(
        input_ptr + row_start + offsets,
        mask=offsets < q_lora_rank,
        other=0.0,
    ).to(tl.float32)

    # RMSNorm
    q_var = tl.sum(q * q, axis=0) / q_lora_rank
    q_rstd = tl.rsqrt(q_var + eps)

    q_norm = q * q_rstd

    q_weight = tl.load(q_weight_ptr + offsets, mask=offsets < q_lora_rank)

    q_out = q_norm * q_weight

    tl.store(q_ptr + row_pid * q_lora_rank + offsets, q_out, mask=offsets < q_lora_rank)

    # k_nope
    k_offsets = offsets + q_lora_rank

    k = tl.load(input_ptr + row_start + k_offsets, mask=offsets < kv_lora_rank, other=0.0).to(tl.float32)

    # RMSNorm
    k_var = tl.sum(k * k, axis=0) / kv_lora_rank
    k_rstd = tl.rsqrt(k_var + eps)

    k_norm = k * k_rstd

    k_weight = tl.load(k_weight_ptr + offsets, mask=offsets < kv_lora_rank)

    k_out = k_norm * k_weight

    tl.store(k_nope_ptr + row_pid * kv_lora_rank + offsets, k_out, mask=offsets < kv_lora_rank)

    # k_pe
    k_pe_offsets = offsets + q_lora_rank + kv_lora_rank

    k_pe = tl.load(input_ptr + row_start + k_pe_offsets, mask=offsets < qk_rope_head_dim, other=0.0)

    tl.store(k_pe_ptr + row_pid * qk_rope_head_dim + offsets, k_pe, mask=offsets < qk_rope_head_dim)


def split_qk_rmsnorm(
        input,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        eps,
        q_weight,
        k_weight,
):
    batch_size = input.shape[0]
    # total_hidden_size = q_lora_rank + kv_lora_rank
    total_hidden_size = input.shape[-1]
    input = input.contiguous()

    q_output = torch.empty(
        batch_size, q_lora_rank, device=input.device, dtype=input.dtype
    )
    k_nope_output = torch.empty(
        batch_size, kv_lora_rank, device=input.device, dtype=input.dtype
    )
    k_pe_output = torch.empty(
        batch_size, qk_rope_head_dim, device=input.device, dtype=input.dtype
    )

    BLOCK_SIZE = triton.next_power_of_2(max(q_lora_rank, kv_lora_rank, qk_rope_head_dim))
    split_qk_rmsnorm_kernel[(batch_size,)](
        input,
        q_output,
        k_nope_output,
        k_pe_output,
        q_weight,
        k_weight,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        total_hidden_size,
        eps,
        BLOCK_SIZE,
    )

    return q_output, k_nope_output, k_pe_output
