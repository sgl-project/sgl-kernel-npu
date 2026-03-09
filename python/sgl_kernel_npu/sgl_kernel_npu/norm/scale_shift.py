import torch
import triton
import triton.language as tl

triton.jit
def fuse_scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    scale_numel: tl.constexpr,
    shift_numel: tl.constexpr,
    block_l: tl.constexpr,
    block_c: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    token_offsets = row_pid * block_l + tl.arange(0, block_l)
    dim_offsets = col_pid * block_c + tl.arange(0, block_c)

    mask = (token_offsets[:, None] < num_tokens) & (dim_offsets[None, :] < hidden_size)
    offset = token_offsets[:, None] * hidden_size + dim_offsets[None, :]

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    if scale_numel == 1:
        scale = tl.load(scale_ptr)
    else:
        scale_offsets = dim_offsets[None, :]
        scale_mask = dim_offsets[None, :] < hidden_size
        scale = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=0.0)

    if shift_numel == 1:
        shift = tl.load(shift_ptr)
    else:
        shift_offsets = dim_offsets[None, :]
        shift_mask = dim_offsets[None, :] < hidden_size
        shift = tl.load(shift_ptr + shift_offsets, mask=shift_mask, other=0.0).to(tl.float32)

    output = x * (1.0 + scale) + shift

    tl.store(
        output_ptr + offset,
        output.to(output_ptr.dtype.element_ty),
        mask=mask
    )


def fuse_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    block_l: int = 128,
    block_c: int = 128,
):
    orig_shape = x.shape
    num_tokens = orig_shape[0] * orig_shape[1]
    hidden_size = orig_shape[2]

    scale = scale.view(-1)
    shift = shift.view(-1)

    scale_numel = scale.numel()
    shift_numel = shift.numel()

    assert scale_numel == 1 or scale_numel == hidden_size, f"Scale size must be 1 or {hidden_size}, got {scale_numel}"
    assert shift_numel == 1 or shift_numel == hidden_size, f"Scale size must be 1 or {hidden_size}, got {shift_numel}"

    output = torch.empty_like(x)

    grid = (
        triton.cdiv(num_tokens, block_l),
        triton.cdiv(hidden_size, block_c),
    )

    fuse_scale_shift_kernel[grid](
        x,
        scale,
        shift,
        output,
        num_tokens,
        hidden_size,
        scale_numel=scale_numel,
        shift_numel=shift_numel,
        block_l=block_l,
        block_c=block_c,
    )

    return output