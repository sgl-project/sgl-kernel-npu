import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _add3_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    offsets = row_start * hidden_size + cols

    for row_idx in tl.range(row_start, n_rows, row_step):
        a = tl.load(a_ptr + offsets, mask=valid_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=valid_mask, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + offsets, mask=valid_mask, other=0.0).to(tl.float32)

        # Preserve eager's two bf16 kernels exactly: the first add is rounded
        # before c participates in the second add.
        ab = (a + b).to(tl.bfloat16)
        output = ab.to(tl.float32) + c
        tl.store(output_ptr + offsets, output, mask=valid_mask)
        offsets += row_step * hidden_size


def add3_bf16_covered(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.shape == b.shape == c.shape
        and a.ndim >= 1
        and a.shape[-1] > 0
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
        and a.device.type == "npu"
    )


def add3_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Return ``bf16(bf16(a + b) + c)`` with one NPU kernel launch."""
    if not add3_bf16_covered(a, b, c):
        raise ValueError("add3_bf16 requires same-shape contiguous NPU bf16 tensors")

    _, num_vectorcore = get_device_properties()
    hidden_size = a.shape[-1]
    n_rows = a.numel() // hidden_size
    block_size = triton.next_power_of_2(hidden_size)
    grid_rows = min(n_rows, num_vectorcore)
    output = torch.empty_like(a)
    _add3_bf16_kernel[(grid_rows, 1, 1)](
        a,
        b,
        c,
        output,
        n_rows,
        hidden_size,
        block_size,
    )
    return output
