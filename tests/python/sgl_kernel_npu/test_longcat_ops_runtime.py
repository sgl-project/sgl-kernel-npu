import os

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


def _get_data_from_pa_cache(key: torch.Tensor, block_table: torch.Tensor, act_s2: int) -> torch.Tensor:
    block_num, block_size, n2, d = key.shape
    assert n2 == 1
    need_block_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_block_num * block_size
    out = torch.zeros((act_s2_align, d), dtype=key.dtype)
    for i in range(need_block_num):
        out[i * block_size : (i + 1) * block_size, :] = key[block_table[i]].reshape(block_size, d)
    return out[:act_s2, :]


def _reference_mlp_lightning_indexer_tnd_pa_bsnd(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    cur_seq_lengths_query: torch.Tensor,
    cur_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    sparse_count: int,
    sparse_mode: int,
):
    n2 = key.shape[2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    t = query.shape[0]
    out_idx = torch.full((t, n2, sparse_count), -1, dtype=torch.int32)
    out_val = torch.zeros((t, n2, sparse_count), dtype=query.dtype)

    batch_size = cur_seq_lengths_query.numel() - 1
    for batch_id in range(batch_size):
        q_start = int(cur_seq_lengths_query[batch_id].item())
        q_end = int(cur_seq_lengths_query[batch_id + 1].item())
        k_start = int(cur_seq_lengths_key[batch_id].item())
        k_end = int(cur_seq_lengths_key[batch_id + 1].item())
        act_s1 = q_end - q_start
        act_s2 = k_end - k_start

        now_q = query[q_start:q_end].transpose(0, 1).to(torch.float32)
        now_weights = weights[q_start:q_end].transpose(0, 1).unsqueeze(-1).to(torch.float32)
        now_k = _get_data_from_pa_cache(key, block_table[batch_id], act_s2).transpose(0, 1).to(torch.float32)

        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0.0, dtype=torch.float32))
        weighted = relu_out * now_weights
        reduced = torch.sum(weighted, dim=0)
        if sparse_mode == 3:
            for i in range(act_s1):
                reduced[-1 - i, act_s2 - i :] = float("-inf")

        topk_values, topk_indices = torch.topk(reduced, k=sparse_count, dim=1, largest=True, sorted=True)
        out_idx[q_start:q_end, 0, :] = topk_indices.to(torch.int32)
        out_val[q_start:q_end, 0, :] = topk_values.to(query.dtype)

    return out_idx, out_val


def _build_multi_batch_pa_case(
    q_lens: list[int],
    k_lens: list[int],
    dtype: torch.dtype,
    n1: int = 8,
    n2: int = 1,
    d: int = 128,
    block_size: int = 16,
):
    assert len(q_lens) == len(k_lens)
    assert all(k_len % block_size == 0 for k_len in k_lens)

    cur_seq_lengths_query = torch.tensor(
        [0, *torch.tensor(q_lens, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int64,
    )
    cur_seq_lengths_key = torch.tensor(
        [0, *torch.tensor(k_lens, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int64,
    )

    t = sum(q_lens)
    total_blocks = sum(k_len // block_size for k_len in k_lens)
    max_blocks = max(k_len // block_size for k_len in k_lens)

    query = torch.randn((t, n1, d), dtype=dtype)
    key = torch.randn((total_blocks, block_size, n2, d), dtype=dtype)
    weights = torch.randn((t, n1), dtype=torch.float32)

    block_table = torch.full((len(q_lens), max_blocks), -1, dtype=torch.int32)
    block_cursor = 0
    for batch_id, k_len in enumerate(k_lens):
        block_num = k_len // block_size
        block_table[batch_id, :block_num] = torch.arange(
            block_cursor, block_cursor + block_num, dtype=torch.int32
        )
        block_cursor += block_num

    return query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key, block_table


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_mlp_lightning_indexer_runtime_tnd_pa_bsnd_matches_reference(dtype: torch.dtype):
    torch.manual_seed(0)

    q_lens = [1, 2]
    k_lens = [16, 24]
    cur_seq_lengths_query = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int64)
    cur_seq_lengths_key = torch.tensor([0, k_lens[0], sum(k_lens)], dtype=torch.int64)

    t = sum(q_lens)
    n1 = 8
    n2 = 1
    d = 128
    block_size = 16
    sparse_count = 8
    sparse_mode = 0

    query = torch.randn((t, n1, d), dtype=dtype)
    key = torch.randn((3, block_size, n2, d), dtype=dtype)
    weights = torch.randn((t, n1), dtype=torch.float32)
    block_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32)

    expected_indices, expected_values = _reference_mlp_lightning_indexer_tnd_pa_bsnd(
        query=query.cpu(),
        key=key.cpu(),
        weights=weights.cpu(),
        cur_seq_lengths_query=cur_seq_lengths_query,
        cur_seq_lengths_key=cur_seq_lengths_key,
        block_table=block_table,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )

    actual_indices, actual_values = torch.ops.npu.mlp_lightning_indexer(
        query.to(DEVICE),
        key.to(DEVICE),
        weights.to(DEVICE),
        cur_seq_lengths_query=cur_seq_lengths_query.to(DEVICE),
        cur_seq_lengths_key=cur_seq_lengths_key.to(DEVICE),
        block_table=block_table.to(DEVICE),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        kv_block_len=1,
        q_block_len=1,
        init_num=0,
        local_num=0,
        sparse_mode=sparse_mode,
        return_value=True,
    )
    actual_indices = actual_indices.cpu()
    actual_values = actual_values.cpu().float()

    assert actual_indices.dtype == torch.int32
    assert actual_values.shape == expected_values.shape
    assert torch.equal(actual_indices, expected_indices)
    torch.testing.assert_close(actual_values, expected_values.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_mlp_lightning_indexer_runtime_tnd_pa_bsnd_sparse_mode_3_matches_reference(dtype: torch.dtype):
    torch.manual_seed(1)

    q_lens = [2, 3]
    k_lens = [16, 32]
    cur_seq_lengths_query = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int64)
    cur_seq_lengths_key = torch.tensor([0, k_lens[0], sum(k_lens)], dtype=torch.int64)

    t = sum(q_lens)
    n1 = 8
    n2 = 1
    d = 128
    block_size = 16
    sparse_count = 8
    sparse_mode = 3

    query = torch.randn((t, n1, d), dtype=dtype)
    key = torch.randn((3, block_size, n2, d), dtype=dtype)
    weights = torch.randn((t, n1), dtype=torch.float32)
    block_table = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32)

    expected_indices, expected_values = _reference_mlp_lightning_indexer_tnd_pa_bsnd(
        query=query.cpu(),
        key=key.cpu(),
        weights=weights.cpu(),
        cur_seq_lengths_query=cur_seq_lengths_query,
        cur_seq_lengths_key=cur_seq_lengths_key,
        block_table=block_table,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )

    actual_indices, actual_values = torch.ops.npu.mlp_lightning_indexer(
        query.to(DEVICE),
        key.to(DEVICE),
        weights.to(DEVICE),
        cur_seq_lengths_query=cur_seq_lengths_query.to(DEVICE),
        cur_seq_lengths_key=cur_seq_lengths_key.to(DEVICE),
        block_table=block_table.to(DEVICE),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        kv_block_len=1,
        q_block_len=1,
        init_num=0,
        local_num=0,
        sparse_mode=sparse_mode,
        return_value=True,
    )
    actual_indices = actual_indices.cpu()
    actual_values = actual_values.cpu().float()

    assert actual_indices.dtype == torch.int32
    assert actual_values.shape == expected_values.shape
    assert torch.equal(actual_indices, expected_indices)
    torch.testing.assert_close(actual_values, expected_values.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_mlp_lightning_indexer_runtime_tnd_pa_bsnd_sparse_mode_3_bs8_matches_reference(dtype: torch.dtype):
    torch.manual_seed(2)

    q_lens = [1, 2, 1, 3, 2, 4, 1, 2]
    k_lens = [16, 32, 48, 64, 80, 96, 112, 128]
    sparse_count = 16
    sparse_mode = 3

    (
        query,
        key,
        weights,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        block_table,
    ) = _build_multi_batch_pa_case(q_lens=q_lens, k_lens=k_lens, dtype=dtype)

    expected_indices, expected_values = _reference_mlp_lightning_indexer_tnd_pa_bsnd(
        query=query.cpu(),
        key=key.cpu(),
        weights=weights.cpu(),
        cur_seq_lengths_query=cur_seq_lengths_query,
        cur_seq_lengths_key=cur_seq_lengths_key,
        block_table=block_table,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )

    actual_indices, actual_values = torch.ops.npu.mlp_lightning_indexer(
        query.to(DEVICE),
        key.to(DEVICE),
        weights.to(DEVICE),
        cur_seq_lengths_query=cur_seq_lengths_query.to(DEVICE),
        cur_seq_lengths_key=cur_seq_lengths_key.to(DEVICE),
        block_table=block_table.to(DEVICE),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        kv_block_len=1,
        q_block_len=1,
        init_num=0,
        local_num=0,
        sparse_mode=sparse_mode,
        return_value=True,
    )
    actual_indices = actual_indices.cpu()
    actual_values = actual_values.cpu().float()

    assert actual_indices.dtype == torch.int32
    assert actual_values.shape == expected_values.shape
    assert torch.equal(actual_indices, expected_indices)
    torch.testing.assert_close(actual_values, expected_values.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_mlp_lightning_indexer_runtime_tnd_pa_bsnd_sparse_mode_3_bs8_sglang_repro(dtype: torch.dtype):
    torch.manual_seed(3)

    q_lens = [32, 48, 16, 64, 24, 40, 56, 72]
    k_lens = [128, 256, 192, 384, 320, 448, 512, 640]
    sparse_count = 128
    init_num = 16
    local_num = 64

    (
        query,
        key,
        weights,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        block_table,
    ) = _build_multi_batch_pa_case(q_lens=q_lens, k_lens=k_lens, dtype=dtype)

    actual_indices, actual_values = torch.ops.npu.mlp_lightning_indexer(
        query.to(DEVICE),
        key.to(DEVICE),
        weights.to(DEVICE),
        cur_seq_lengths_query=cur_seq_lengths_query.to(DEVICE),
        cur_seq_lengths_key=cur_seq_lengths_key.to(DEVICE),
        block_table=block_table.to(DEVICE),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        kv_block_len=1,
        q_block_len=1,
        init_num=init_num,
        local_num=local_num,
        sparse_mode=3,
        return_value=True,
    )

    assert actual_indices.shape == (sum(q_lens), 1, sparse_count)
    assert actual_indices.dtype == torch.int32
    assert actual_values.shape == (sum(q_lens), 1, sparse_count)
