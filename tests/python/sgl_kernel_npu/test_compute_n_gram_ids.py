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


def _build_oe_tables(num_embeddings: int, oe_n: int, oe_k: int, over_embedding_m: int):
    oe_mods = torch.zeros((oe_n - 1, oe_k), dtype=torch.int32)
    oe_weights = torch.zeros((oe_n - 1, oe_k, oe_n), dtype=torch.int32)
    exclusive = torch.zeros(((oe_n - 1) * oe_k + 1,), dtype=torch.int32)
    running = 0
    flat_idx = 0
    for n in range(2, oe_n + 1):
        for k in range(oe_k):
            mod = over_embedding_m + 2 * ((n - 2) * oe_k + k) + 1
            oe_mods[n - 2, k] = mod
            exclusive[flat_idx] = running
            for delta in range(oe_n):
                oe_weights[n - 2, k, delta] = pow(num_embeddings, delta, mod)
            running += mod
            flat_idx += 1
    exclusive[flat_idx] = running
    return oe_weights, oe_mods, exclusive


def _reference_compute_n_gram_ids(
    oe_weights: torch.Tensor,
    oe_mods: torch.Tensor,
    exclusive_oe_embeder_size_sums: torch.Tensor,
    tokens: torch.Tensor,
    exclusive_req_len_sums: torch.Tensor,
    oe_token_table: torch.Tensor,
    row_indices: torch.Tensor,
    column_starts: torch.Tensor,
    batch_size: int,
    oe_n: int,
    oe_k: int,
    max_context_len: int,
) -> torch.Tensor:
    out = torch.empty((tokens.numel(), (oe_n - 1) * oe_k), dtype=torch.int32)
    for req_id in range(batch_size):
        start = 0 if req_id == 0 else int(exclusive_req_len_sums[req_id - 1].item())
        end = int(exclusive_req_len_sums[req_id].item())
        req_row = int(row_indices[req_id].item())
        req_base = req_row * max_context_len
        req_cur = req_base + int(column_starts[req_id].item())
        for token_idx in range(start, end):
            current_token_offset = token_idx - start
            current_token_table_index = req_cur + current_token_offset
            for n in range(oe_n - 1):
                for k in range(oe_k):
                    oe_mod = int(oe_mods[n, k].item())
                    n_gram_id = 0
                    for j in range(n + 2):
                        if current_token_table_index - j < req_base:
                            break
                        token = int(oe_token_table.view(-1)[current_token_table_index - j].item())
                        if token < 0:
                            break
                        weight = int(oe_weights[n, k, j].item())
                        n_gram_id += (token * weight) % oe_mod
                    n_gram_id %= oe_mod
                    n_gram_id += int(exclusive_oe_embeder_size_sums[n * oe_k + k].item())
                    out[token_idx, n * oe_k + k] = n_gram_id
    return out


def test_compute_n_gram_ids_runtime_matches_reference():
    num_embeddings = 32
    oe_n = 3
    oe_k = 2
    over_embedding_m = 5
    max_context_len = 6

    oe_weights, oe_mods, exclusive_oe_embeder_size_sums = _build_oe_tables(
        num_embeddings=num_embeddings,
        oe_n=oe_n,
        oe_k=oe_k,
        over_embedding_m=over_embedding_m,
    )
    tokens = torch.tensor([8, 9, 4, 5], dtype=torch.int32)
    exclusive_req_len_sums = torch.tensor([2, 4], dtype=torch.int32)
    oe_token_table = torch.tensor(
        [
            [3, 4, 5, -1, -1, -1],
            [7, 8, 9, 10, -1, -1],
        ],
        dtype=torch.int32,
    )
    row_indices = torch.tensor([1, 0], dtype=torch.int64)
    column_starts = torch.tensor([1, 1], dtype=torch.int32)

    expected = _reference_compute_n_gram_ids(
        oe_weights=oe_weights,
        oe_mods=oe_mods,
        exclusive_oe_embeder_size_sums=exclusive_oe_embeder_size_sums,
        tokens=tokens,
        exclusive_req_len_sums=exclusive_req_len_sums,
        oe_token_table=oe_token_table,
        row_indices=row_indices,
        column_starts=column_starts,
        batch_size=2,
        oe_n=oe_n,
        oe_k=oe_k,
        max_context_len=max_context_len,
    )

    actual = torch.ops.npu.compute_n_gram_ids(
        oe_weights.to(DEVICE),
        oe_mods.to(DEVICE),
        exclusive_oe_embeder_size_sums.to(DEVICE),
        tokens.to(DEVICE),
        exclusive_req_len_sums.to(DEVICE),
        oe_token_table.to(DEVICE),
        row_indices.to(DEVICE),
        column_starts.to(DEVICE),
        batch_size=2,
        oe_n=oe_n,
        oe_k=oe_k,
        max_context_len=max_context_len,
    ).cpu()

    assert actual.dtype == torch.int32
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
