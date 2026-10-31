"""Independent PyTorch triangular solves for chunk-local (I + A)^-1."""

import torch


def _validate_inputs(A, cu_seqlens, output_dtype):
    assert A.ndim == 4, f"A must have shape [B, T, H, C], got {A.shape}"
    assert all(n > 0 for n in A.shape), f"A must be nonempty, got {A.shape}"
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert A.dtype in dtypes, f"Unsupported A dtype: {A.dtype}"
    assert output_dtype in dtypes, f"Unsupported output_dtype: {output_dtype}"
    if cu_seqlens is not None:
        assert A.shape[0] == 1, "Packed variable-length input requires B=1"
        assert (
            cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2
        ), "cu_seqlens must be a 1-D tensor with at least two entries"
        assert cu_seqlens.dtype in (
            torch.int32,
            torch.int64,
        ), f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}"
    # Strict triangularity and finite values remain caller contracts.


def solve_tril(A, cu_seqlens=None, output_dtype=torch.float):
    _validate_inputs(A, cu_seqlens, output_dtype)
    B, T, H, C = A.shape
    boundaries = [0, T] if cu_seqlens is None else cu_seqlens.tolist()
    if cu_seqlens is not None:
        assert boundaries[0] == 0, "cu_seqlens must start at 0"
        assert boundaries[-1] == T, f"cu_seqlens must end at T={T}"
        assert all(
            bos < eos for bos, eos in zip(boundaries[:-1], boundaries[1:])
        ), "cu_seqlens must be strictly increasing"
    result = torch.empty_like(A, dtype=output_dtype)
    identity = torch.eye(C, dtype=torch.float32, device=A.device)
    for batch in range(B):
        for bos, eos in zip(boundaries[:-1], boundaries[1:]):
            length = eos - bos
            padding = (-length) % C
            # [L, H, C] -> [H, number_of_chunks, C, C]. Padding gives
            # incomplete chunks an identity extension before the solve.
            rows = A[batch, bos:eos].float()
            if padding:
                rows = torch.cat([rows, rows.new_zeros(padding, H, C)], dim=0)
            blocks = rows.reshape(-1, C, H, C).permute(2, 0, 1, 3)
            inverse = torch.linalg.solve_triangular(
                identity + blocks,
                identity.expand_as(blocks),
                upper=False,
                unitriangular=True,
            )
            rows = inverse.permute(1, 2, 0, 3).reshape(-1, H, C)
            result[batch, bos:eos] = rows[:length].to(output_dtype)
    return result
