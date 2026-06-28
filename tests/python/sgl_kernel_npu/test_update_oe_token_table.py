import pytest
import torch


try:
    import sgl_kernel_npu  # noqa: F401
    import torch_npu
except (ImportError, OSError):
    pytest.skip("sgl_kernel_npu/torch_npu not available", allow_module_level=True)


if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("NPU not available", allow_module_level=True)


DEVICE = "npu:0"
torch_npu.npu.set_device(0)


def _reference_update_oe_token_table(
    oe_token_table: torch.Tensor,
    tokens: torch.Tensor,
    req_lens: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    ignore_tokens: torch.Tensor,
) -> torch.Tensor:
    out = oe_token_table.clone()
    ignore_set = set(ignore_tokens.to(torch.int32).cpu().tolist())
    src_offset = 0
    for req_id in range(req_lens.numel()):
        req_len = int(req_lens[req_id].item())
        row = int(row_indices[req_id].item())
        col_start = int(column_starts[req_id].item())
        if req_len <= 0:
            continue
        values = tokens[src_offset : src_offset + req_len].to(torch.int32).clone()
        if ignore_set:
            for idx in range(values.numel()):
                if int(values[idx].item()) in ignore_set:
                    values[idx] = -1
        out[row, col_start : col_start + req_len] = values
        src_offset += req_len
    return out


def test_update_oe_token_table_runtime_matches_reference_without_ignore_tokens():
    oe_token_table = torch.full((4, 8), -1, dtype=torch.int32)
    tokens = torch.tensor([11, 12, 13, 21, 22], dtype=torch.int32)
    req_lens = torch.tensor([3, 0, 2], dtype=torch.int32)
    row_indices = torch.tensor([2, 1, 0], dtype=torch.int64)
    column_starts = torch.tensor([1, 4, 3], dtype=torch.int32)
    ignore_tokens = torch.empty((0,), dtype=torch.int32)

    expected = _reference_update_oe_token_table(
        oe_token_table=oe_token_table,
        tokens=tokens,
        req_lens=req_lens,
        row_indices=row_indices,
        column_starts=column_starts,
        ignore_tokens=ignore_tokens,
    )

    actual_table = oe_token_table.clone().to(DEVICE)
    actual = torch.ops.npu.update_oe_token_table(
        tokens.to(DEVICE),
        req_lens.to(DEVICE),
        row_indices.to(DEVICE),
        column_starts.to(DEVICE),
        ignore_tokens.to(DEVICE),
        batch_size=req_lens.numel(),
        max_context_len=actual_table.shape[1],
        oe_token_table=actual_table,
    ).cpu()

    assert actual.dtype == torch.int32
    assert torch.equal(actual, expected)
    assert torch.equal(actual_table.cpu(), expected)


def test_update_oe_token_table_runtime_matches_reference_with_ignore_tokens():
    oe_token_table = torch.full((3, 7), -7, dtype=torch.int32)
    tokens = torch.tensor([5, 9, 6, 7, 9, 8], dtype=torch.int32)
    req_lens = torch.tensor([2, 4], dtype=torch.int32)
    row_indices = torch.tensor([1, 2], dtype=torch.int64)
    column_starts = torch.tensor([0, 2], dtype=torch.int32)
    ignore_tokens = torch.tensor([9, 10], dtype=torch.int32)

    expected = _reference_update_oe_token_table(
        oe_token_table=oe_token_table,
        tokens=tokens,
        req_lens=req_lens,
        row_indices=row_indices,
        column_starts=column_starts,
        ignore_tokens=ignore_tokens,
    )

    actual_table = oe_token_table.clone().to(DEVICE)
    actual = torch.ops.npu.update_oe_token_table(
        tokens.to(DEVICE),
        req_lens.to(DEVICE),
        row_indices.to(DEVICE),
        column_starts.to(DEVICE),
        ignore_tokens.to(DEVICE),
        batch_size=req_lens.numel(),
        max_context_len=actual_table.shape[1],
        oe_token_table=actual_table,
    ).cpu()

    assert actual.dtype == torch.int32
    assert torch.equal(actual, expected)
    assert torch.equal(actual_table.cpu(), expected)
