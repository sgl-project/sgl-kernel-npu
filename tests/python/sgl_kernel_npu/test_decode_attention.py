import torch
import triton
import triton.language as tl
from sgl_kernel_npu.attention import decode_gqa, decode_gqa_high_performance, decode_mla


def get_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    else:
        return torch.device("cpu")


def decode_gqa_golden(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    q_head_num = query.shape[1]
    _, block_size, num_kv_heads, k_head_dim = key_cache.shape
    v_head_dim = value_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, k_head_dim)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_dim)[:kv_len]

        if q_head_num != num_kv_heads:
            assert (
                q_head_num % num_kv_heads == 0
            ), "q_head_num must be divisible by num_kv_heads"
            k = torch.repeat_interleave(k, q_head_num // num_kv_heads, dim=1)
            v = torch.repeat_interleave(v, q_head_num // num_kv_heads, dim=1)
        qk = torch.einsum("qhd,khd->hqk", q, k).float()
        score = torch.softmax(qk, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", score, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def test_gqa(B, S, H_Q, H_KV, D_Q, D_V, dtype):
    device = get_device()
    torch.manual_seed(1)
    seq_lens = torch.full((B,), S, device=device, dtype=torch.int32)
    page_size = 128
    max_page_num = (S + page_size - 1) // page_size

    q = torch.randn((B, H_Q, D_Q), device=device, dtype=dtype)
    k_buffer = torch.randn(
        (max_page_num * B, page_size, H_KV, D_Q), device=device, dtype=dtype
    )
    v_buffer = k_buffer[..., :D_V]
    block_table = torch.arange(
        0, B * max_page_num, device=device, dtype=torch.int32
    ).reshape(B, max_page_num)

    attn_logits = torch.empty((B, H_Q, D_V), device=device, dtype=dtype)
    sm_scale = 1.0 / (D_Q**0.5)
    decode_gqa(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        seq_lens,
        sm_scale,
        page_size,
        block_table,
    )
    torch.npu.synchronize()

    attn_logits1 = torch.empty((B, H_Q, D_V), device=device, dtype=dtype)
    qk_out = torch.randn((B, H_Q, S), device=device, dtype=dtype)
    p_ptr = torch.randn((B, H_Q, S), device=device, dtype=dtype)
    pv_ptr = torch.randn((B, H_Q, D_V), device=device, dtype=dtype)
    decode_gqa_high_performance(
        q,
        k_buffer,
        v_buffer,
        attn_logits1,
        seq_lens,
        qk_out,
        p_ptr,
        pv_ptr,
        sm_scale,
        page_size,
        block_table,
    )
    torch.npu.synchronize()

    q_len = B * [1]
    kv_len = seq_lens.cpu()
    attn_logits2 = decode_gqa_golden(
        q, k_buffer, v_buffer, q_len, kv_len, block_table, sm_scale
    )
    torch.npu.synchronize()

    print(
        "Max diff of gpq vs golden: ", torch.max(torch.abs(attn_logits - attn_logits2))
    )
    assert torch.allclose(attn_logits, attn_logits2, rtol=1e-2, atol=1e-2)

    print(
        "Max diff of high-perf gpq vs golden: ",
        torch.max(torch.abs(attn_logits1 - attn_logits2)),
    )
    assert torch.allclose(attn_logits1, attn_logits2, rtol=1e-2, atol=1e-2)


def decode_mla_golden(
    query: torch.Tensor,
    key_cache_nope: torch.Tensor,
    value_cache: torch.Tensor,
    key_cache_rope: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    q_head_num = query.shape[1]
    _, block_size, num_kv_heads, qk_nope_dim = key_cache_nope.shape
    qk_rope_dim = key_cache_rope.shape[-1]
    v_head_dim = value_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        # Split Q into Q_nope and Q_rope
        q_nope = q[:, :, :qk_nope_dim]
        q_rope = q[:, :, qk_nope_dim:]

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k_nope = key_cache_nope[block_indices].view(-1, num_kv_heads, qk_nope_dim)[
            :kv_len
        ]
        k_rope = key_cache_rope[block_indices].view(-1, num_kv_heads, qk_rope_dim)[
            :kv_len
        ]
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_dim)[:kv_len]

        if q_head_num != num_kv_heads:
            assert (
                q_head_num % num_kv_heads == 0
            ), "q_head_num must be divisible by num_kv_heads"
            rep_factor = q_head_num // num_kv_heads
            k_nope = torch.repeat_interleave(k_nope, rep_factor, dim=1)
            k_rope = torch.repeat_interleave(k_rope, rep_factor, dim=1)
            v = torch.repeat_interleave(v, rep_factor, dim=1)

        qk_nope = torch.einsum("qhd,khd->hqk", q_nope, k_nope).float()
        qk_rope = torch.einsum("qhd,khd->hqk", q_rope, k_rope).float()
        qk = (qk_nope + qk_rope) * scale
        score = torch.softmax(qk, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", score, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def test_mla(B, S, H_Q, H_KV, D_Q, D_V, dtype):
    device = get_device()
    torch.manual_seed(2)
    seq_lens = torch.full((B,), S, device=device, dtype=torch.int32)
    page_size = 128
    max_page_num = (S + page_size - 1) // page_size

    q = torch.randn((B, H_Q, D_Q), device=device, dtype=dtype)
    k_nope = torch.randn(
        (max_page_num * B, page_size, H_KV, D_V), device=device, dtype=dtype
    )
    k_rope = torch.randn(
        (max_page_num * B, page_size, H_KV, D_Q - D_V), device=device, dtype=dtype
    )
    block_table = torch.arange(
        0, B * max_page_num, device=device, dtype=torch.int32
    ).reshape(B, max_page_num)

    attn_logits = torch.empty((B, H_Q, D_V), device=device, dtype=dtype)
    sm_scale = 1.0 / (D_Q**0.5)

    decode_mla(
        q,
        k_nope,
        k_rope,
        attn_logits,
        seq_lens,
        sm_scale,
        page_size,
        block_table,
    )
    torch.npu.synchronize()

    q_len = B * [1]
    kv_len = seq_lens.cpu()
    attn_logits1 = decode_mla_golden(
        q, k_nope, k_nope, k_rope, q_len, kv_len, block_table, sm_scale
    )
    torch.npu.synchronize()

    print(
        "Max diff of mla vs golden: ", torch.max(torch.abs(attn_logits - attn_logits1))
    )
    assert torch.allclose(attn_logits, attn_logits1, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    dtypes = [torch.bfloat16, torch.float16]
    seq_lens = [4096, 3589, 1314, 128]
    gqa_configs = [
        (1, 32, 1, 576, 512),
        (16, 128, 1, 576, 512),
        (16, 128, 1, 288, 256),
        (16, 64, 8, 128, 128),
    ]

    for dtype in dtypes:
        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in gqa_configs:
                test_gqa(B, S, H_Q, H_KV, D, D_V, dtype)

    mla_configs = [
        (16, 8, 1, 576, 512),
        (16, 32, 1, 576, 512),
        (16, 64, 1, 576, 512),
        (16, 128, 1, 576, 512),
    ]
    for dtype in dtypes:
        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in mla_configs:
                test_mla(B, S, H_Q, H_KV, D, D_V, dtype)
