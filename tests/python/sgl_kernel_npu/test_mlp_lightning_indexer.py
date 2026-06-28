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
def test_mlp_lightning_indexer_runtime_tnd_pa_bsnd_sparse_mode_3_bs8_matches_reference(
    dtype: torch.dtype,
):
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
