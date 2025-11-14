
import triton
import triton.language as tl
import torch


@triton.jit
def _paged_mla_fwd_kernel(
    Q,
    K_NOPE_Buffer,
    K_ROPE_Buffer,
    sm_scale,
    kv_seq_lens,
    Att_Out,
    block_table,
    stride_block_table_batch: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_knbs: tl.constexpr, 
    stride_buf_knpage: tl.constexpr,
    stride_buf_knh: tl.constexpr,
    stride_buf_krbs: tl.constexpr,
    stride_buf_krpage: tl.constexpr,
    stride_buf_krh: tl.constexpr,
    stride_mid_ob: tl.constexpr,
    stride_mid_oh: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lkv: tl.constexpr,
    Lrope: tl.constexpr,
):
    """
    Forward kernel for Multi-Latent Attention (MLA) with paged KV cache support.
    
    Support bfloat16 and float16

    MLA separates query/key into two components:
        - K/Q_nope: Content-aware features without rotation.
        - K/Q_rope: Positional features with RoPE applied.

    The attention score is computed as:
        score = (Q_nope @ K_nope.T) + (Q_rope @ K_rope.T)

    Memory layout follows paged attention:
        - KV caches are stored in blocks of shape [page_size, num_heads, head_dim].
        - `block_table` maps logical pages to physical block IDs in NPU memory.

    This kernel computes one group of query heads per launch.

    Args:
        Q (Tensor): Input queries of shape [batch_size, q_head_num, D], where D = Lkv + Lrope.
        K_NOPE_Buffer (Tensor): Paged key buffer (non-rotated) of shape [num_blocks, page_size, kv_head_num, Lkv].
        K_ROPE_Buffer (Tensor): Paged key buffer (RoPE-applied) of shape [num_blocks, page_size, kv_head_num, Lrope].
        sm_scale (float): Scaling factor for attention scores (usually 1/sqrt(head_dim)).
        kv_seq_lens (Tensor): Current sequence lengths for each batch item, shape [batch_size].
        Att_Out (Tensor): Output buffer of shape [batch_size, q_head_num, Lkv].
        block_table (Tensor): Mapping from logical page index to physical block ID, shape [batch_size, max_pages].
        ... (strides): Precomputed strides for efficient indexing.
        q_heads_per_kv_head (int): Number of query heads sharing a single KV head (q_head_num // kv_head_num).
        q_head_num (int): Total number of query heads.
        BLOCK_DNOPE (int): Tiling size for nope dimension (padded to power-of-2).
        BLOCK_DROPE (int): Tiling size for rope dimension (padded to power-of-2).
        BLOCK_N (int): Page size (number of tokens per kv cache block).
        BLOCK_H (int): Number of query heads processed per block.
        Lkv (int): Actual dimension of K/Q_nope.
        Lrope (int): Actual dimension of K/Q_rope.
    """

    cur_batch = tl.program_id(0)
    cur_head_group_id = tl.program_id(1)
    # Determine how many query heads this block processes
    cur_kv_head = cur_head_group_id // tl.cdiv(q_heads_per_kv_head, BLOCK_H)
    if BLOCK_H < q_heads_per_kv_head:
        GROUP_HEAD_NUM: tl.constexpr = BLOCK_H
    else:
        GROUP_HEAD_NUM: tl.constexpr = q_heads_per_kv_head

    # Step 1: Load Q_nope (content part)
    offset_h = cur_head_group_id * GROUP_HEAD_NUM + tl.arange(0, GROUP_HEAD_NUM)
    mask_h = offset_h < q_head_num
    offset_qk_nope = tl.arange(0, BLOCK_DNOPE)
    mask_qk_nope = offset_qk_nope < Lkv
    offset_q = cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_qk_nope[None, :]
    q = tl.load(Q + offset_q, mask=(mask_h[:, None]) & (mask_qk_nope[None, :]))

    # Step 2: Load Q_rope (positional part)
    offset_qk_rope = tl.arange(0, BLOCK_DROPE)
    mask_qk_rope = offset_qk_rope < Lrope
  
    # RoPE part starts after nope part in Q
    offset_qrope = BLOCK_DNOPE + offset_qk_rope
    mask_qrope = offset_qrope < (Lkv + Lrope)
    offset_qpe = cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_qrope[None, :]
    q_pe = tl.load(Q + offset_qpe, mask=(mask_h[:, None]) & (mask_qrope[None, :]))

    # Step 3: Iterate over physical blocks using PagedAttention
    cur_seq_len = tl.load(kv_seq_lens + cur_batch)
    page_num = tl.cdiv(cur_seq_len, BLOCK_N)
    offset_page = tl.arange(0, BLOCK_N)

    history_max = tl.zeros([GROUP_HEAD_NUM], dtype=tl.float32) - float('inf')
    l = tl.zeros([GROUP_HEAD_NUM], dtype=tl.float32)
    acc = tl.zeros([GROUP_HEAD_NUM, BLOCK_DNOPE], dtype=tl.float32)
    for page_id in range(page_num):
        # --- Load K_nope ---
        page_loc = tl.load(block_table + cur_batch * stride_block_table_batch + page_id)
        offset_k = (page_loc * stride_buf_knbs 
                    + offset_page[:, None] * stride_buf_knpage 
                    + cur_kv_head * stride_buf_knh 
                    + offset_qk_nope[None, :])
        mask_page = (page_id * BLOCK_N + offset_page) < cur_seq_len
        k = tl.load(K_NOPE_Buffer + offset_k, mask=(mask_page[:, None] & mask_qk_nope[None, :]))
        v = k # In MLA, V shares same buffer as K_nope

        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)

        # --- Load K_rope ---
        offset_krope = (page_loc * stride_buf_krbs 
                        + offset_page[:, None] * stride_buf_krpage 
                        + cur_kv_head * stride_buf_krh 
                        + offset_qk_rope[None, :])
        k_pe = tl.load(K_ROPE_Buffer + offset_krope, mask=(mask_page[:, None] & mask_qk_rope[None, :]))
        k_pe = tl.trans(k_pe, (1, 0))
        # Combine both parts
        qk += tl.dot(q_pe, k_pe)

        qk = qk * sm_scale
        # Online softmax update 
        qk = tl.where((mask_h[:, None] & mask_page[None, :]), qk, float("-inf"))
        new_e_max = tl.maximum(tl.max(qk, 1), history_max)
        re_scale = tl.exp(history_max - new_e_max)
        p_exp = tl.exp(qk - new_e_max[:, None])

        l = l * re_scale + tl.sum(p_exp, 1)
        acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
        history_max = new_e_max

    offs_mid_o = (cur_batch * stride_mid_ob
                    + offset_h[:, None] * stride_mid_oh
                    + offset_qk_nope[None, :])
    tl.store(Att_Out + offs_mid_o, acc / l[:, None], mask=(mask_h[:, None] & mask_qk_nope[None, :]))


def decode_mla(
    q,
    k_nope_buffer,
    k_rope_buffer,
    att_out,
    kv_seq_lens,
    sm_scale,
    page_size,
    block_table,
):
    """
    Python wrapper to launch MLA forward kernel with paged KV cache.

    Args:
        q (Tensor): Queries, shape [batch_size, q_head_num, Lkv + Lrope]
        k_nope_buffer (Tensor): Paged K_nope cache, shape [num_blocks, page_size, kv_head_num, Lkv]
        k_rope_buffer (Tensor): Paged K_rope cache, shape [num_blocks, page_size, kv_head_num, Lrope]
        att_out (Tensor): Output buffer, shape [batch_size, q_head_num, Lkv]
        kv_seq_lens (Tensor): Current sequence lengths, shape [batch_size]
        sm_scale (float): Attention scale (e.g., 1/sqrt(head_dim))
        page_size (int): Number of tokens per KV cache block
        block_table (Tensor): Logical-to-physical block mapping, shape [batch_size, max_blocks]
    """
    Lkv = k_nope_buffer.shape[-1]
    Lrope = k_rope_buffer.shape[-1]
    BLOCK_DROPE = triton.next_power_of_2(Lrope)
    BLOCK_DNOPE = triton.next_power_of_2(Lkv)

    BLOCK_H = 16
    BLOCK_N = page_size
    batch, q_head_num = q.shape[0], q.shape[1]
    kv_head_num = k_nope_buffer.shape[2]
    q_heads_per_kv_head = q_head_num // kv_head_num
    grid = (
        batch,
        triton.cdiv(q_head_num, min(BLOCK_H, q_heads_per_kv_head)),
    )
    _paged_mla_fwd_kernel[grid](
        q,
        k_nope_buffer,
        k_rope_buffer,
        sm_scale,
        kv_seq_lens,
        att_out,
        block_table,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        k_nope_buffer.stride(0),
        k_nope_buffer.stride(1),
        k_nope_buffer.stride(2),
        k_rope_buffer.stride(0),
        k_rope_buffer.stride(1),
        k_rope_buffer.stride(2),
        att_out.stride(0),
        att_out.stride(1),
        q_heads_per_kv_head=q_heads_per_kv_head,
        q_head_num=q_head_num,
        BLOCK_DNOPE=BLOCK_DNOPE,
        BLOCK_DROPE=BLOCK_DROPE,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        Lkv=Lkv,
        Lrope=Lrope,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        multibuffer=False
    )


@triton.jit
def _paged_gqa_fwd_kernel(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_seq_lens,
    Att_Out,
    block_table,
    stride_block_table_batch: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_kbs: tl.constexpr,
    stride_buf_kpage: tl.constexpr,
    stride_buf_kh: tl.constexpr,
    stride_buf_vbs: tl.constexpr,
    stride_buf_vpage: tl.constexpr,
    stride_buf_vh: tl.constexpr,
    stride_mid_ob: tl.constexpr,
    stride_mid_oh: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr
):
    """
    Forward kernel for Grouped-Query Attention (GQA) with paged KV cache.

    GQA allows multiple query heads to share the same key/value heads,
    reducing memory bandwidth while preserving expressiveness.

    Uses online softmax for numerical stability during incremental decoding.

    Args:
        Q (Tensor): Queries, shape [batch_size, q_head_num, Lk].
        K_Buffer (Tensor): Paged key cache, shape [num_blocks, page_size, kv_head_num, Lk].
        V_Buffer (Tensor): Paged value cache, shape [num_blocks, page_size, kv_head_num, Lv].
        sm_scale (float): Attention scaling factor.
        kv_seq_lens (Tensor): Sequence lengths, shape [batch_size].
        Att_Out (Tensor): Output buffer, shape [batch_size, q_head_num, Lv].
        block_table (Tensor): Logical-to-physical block mapping.
        ... (strides): Memory strides.
        q_heads_per_kv_head (int): Ratio of Q heads to KV heads.
        q_head_num (int): Total number of query heads.
        BLOCK_DMODEL (int): Tiled size for key dimension (padded).
        BLOCK_DPE (int): Optional extra dim for RoPE (can be zero).
        BLOCK_DV (int): Tiled size for value dimension (padded).
        BLOCK_N (int): Page size.
        BLOCK_H (int): Number of query heads processed per block.
        Lk (int): Actual key dimension.
        Lv (int): Actual value dimension.
    """
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    if BLOCK_H < q_heads_per_kv_head:
        HEAD_NUM: tl.constexpr = BLOCK_H
    else:
        HEAD_NUM: tl.constexpr = q_heads_per_kv_head
    cur_q_head_start =  cur_kv_head * q_heads_per_kv_head

    for head_block_start in range(0, q_heads_per_kv_head, HEAD_NUM):
        # Step 1: Load Q_nope + Q_rope
        offset_h = cur_q_head_start + head_block_start + tl.arange(0, HEAD_NUM)
        offset_d = tl.arange(0, BLOCK_DMODEL + BLOCK_DPE)
        mask_h = offset_h < q_head_num
        mask_d = offset_d < Lk
        offset_q = cur_batch * stride_qbs + offset_h[:, None] * stride_qh + offset_d[None, :]
        q = tl.load(Q + offset_q, mask=(mask_h[:, None]) & (mask_d[None, :]))

        # Step 2: Iterate over physical blocks using PagedAttention
        cur_seq_len = tl.load(kv_seq_lens + cur_batch)
        page_num = tl.cdiv(cur_seq_len, BLOCK_N)

        offset_page = tl.arange(0, BLOCK_N)
        offset_dv = tl.arange(0, BLOCK_DV)
        mask_dv = offset_dv < Lv

        history_max = tl.zeros([HEAD_NUM], dtype=tl.float32) - float('inf')
        l = tl.zeros([HEAD_NUM], dtype=tl.float32)
        acc = tl.zeros([HEAD_NUM, BLOCK_DV], dtype=tl.float32)
        for page_id in range(page_num):
            # Load K
            page_loc = tl.load(block_table + cur_batch * stride_block_table_batch + page_id)
            offset_k = (page_loc * stride_buf_kbs 
                        + offset_page[:, None] * stride_buf_kpage 
                        + cur_kv_head * stride_buf_kh 
                        + offset_d[None, :])
            mask_page = (page_id * BLOCK_N + offset_page) < cur_seq_len
            k = tl.load(K_Buffer + offset_k, mask=(mask_page[:, None] & mask_d[None, :]))
            k = tl.trans(k, (1, 0))
            qk = tl.dot(q, k)

            # Load V early to overlap with computation
            offset_v = (page_loc * stride_buf_vbs 
                        + offset_page[:, None] * stride_buf_vpage 
                        + cur_kv_head * stride_buf_vh 
                        + offset_dv[None, :])
            v = tl.load(V_Buffer + offset_v, mask=(mask_page[:, None] &mask_dv[None, :]))

            qk = qk * sm_scale
            qk = tl.where((mask_h[:, None] & mask_page[None, :]), qk, float("-inf"))
            new_e_max = tl.maximum(tl.max(qk, 1), history_max)
            re_scale = tl.exp(history_max - new_e_max)
            p_exp = tl.exp(qk - new_e_max[:, None])
            
            # Online softmax update 
            l = l * re_scale + tl.sum(p_exp, 1)
            acc = acc * re_scale[:, None] + tl.dot(p_exp.to(v.dtype), v)
            history_max = new_e_max

        offs_mid_o = (cur_batch * stride_mid_ob
                        + offset_h[:, None] * stride_mid_oh
                        + offset_dv[None, :])
        tl.store(Att_Out + offs_mid_o, acc / l[:, None], mask=(mask_h[:, None] & mask_dv[None, :]))


def decode_gqa(
    q,
    k_buffer,
    v_buffer,
    att_out,
    kv_seq_lens,
    sm_scale,
    page_size,
    block_table,
):
    """
    Wrapper function to launch GQA forward kernel.

    Handles special cases for known architectures (e.g., DeepSeek-V3 uses split K).

    Args:
        q (Tensor): Input queries
        k_buffer (Tensor): Paged key cache
        v_buffer (Tensor): Paged value cache
        att_out (Tensor): Output buffer
        kv_seq_lens (Tensor): Sequence lengths
        sm_scale (float): Attention scale
        page_size (int): Size of each KV block
        block_table (Tensor): Block mapping table
    """
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    BLOCK_N = page_size
    BLOCK_H = 32
    # Special-case tiling for models like DeepSeek-V3 which split K into model+RoPE parts
    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
        BLOCK_H = 16
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, q_head_num = q.shape[0], q.shape[1]
    kv_head_num = k_buffer.shape[2]
    q_heads_per_kv_head = q_head_num // kv_head_num
    assert q_head_num % kv_head_num == 0, "head_num must be divisible by kv_head_num"

    grid = (
        batch,
        kv_head_num
    )
    _paged_gqa_fwd_kernel[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_seq_lens,
        att_out,
        block_table,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        att_out.stride(0),
        att_out.stride(1),
        q_heads_per_kv_head=q_heads_per_kv_head,
        q_head_num=q_head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        Lk=Lk,
        Lv=Lv,
        limit_auto_multi_buffer_only_for_local_buffer=False, 
        multibuffer=True
    )

def get_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        # 华为昇腾 NPU 优先
        return torch.device('npu:0')
    elif torch.cuda.is_available():
        # NVIDIA CUDA
        return torch.device('cuda:0')
    else:
        # 默认使用 CPU
        return torch.device('cpu')


def load_tensors(save_path, device):
    # 1. 加载保存的张量（默认会保留原始设备信息，如 npu:0）
    try:
        loaded_data = torch.load(save_path, map_location="cpu")  # 先加载到 CPU，避免设备不兼容
        print(f"已从 {save_path} 加载数据到 CPU")
    except FileNotFoundError:
        print(f"错误：未找到文件 {save_path}")
        return
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return


    migrated_data = {}
    for key, value in loaded_data.items():
        if isinstance(value, torch.Tensor):
            # 张量迁移到 GPU
            migrated_tensor = value.to(device)
            migrated_data[key] = migrated_tensor
            print(f"张量 {key} 已迁移到 {device}, 设备：{migrated_tensor.device}")
        elif isinstance(value, dict) and "params" in key:
            # 元数据无需迁移，直接保留
            migrated_data[key] = value
        else:
            migrated_data[key] = value

    return migrated_data


def test_grouped_decode_attention_kernel(B, S, H_Q, H_KV, DQ, DV, dtype):
    device = get_device()
    torch.manual_seed(1)
    seq_lens = torch.full((B, ), S, device=device, dtype=torch.int32)
    page_size = 128
    max_page_num = (S + page_size - 1) // page_size

    q = torch.randn((B, H_Q, DQ), device=device, dtype=dtype)
    k_buffer = torch.randn((max_page_num * B, page_size, H_KV, DQ), device=device, dtype=dtype)
    v_buffer = k_buffer[... , : DV]
    block_table = torch.arange(0, B * max_page_num, device=device, dtype=torch.int32).reshape(B, max_page_num)
    
    attn_logits1 = torch.empty((B, H_Q, DV), device=device, dtype=dtype)
    sm_scale = 1.0 / (DQ**0.5)

    for i in range(10):
        decode_gqa(
            q,
            k_buffer,
            v_buffer,
            attn_logits1,
            seq_lens,
            sm_scale,
            page_size,
            block_table,
        )
    torch.npu.synchronize()

    q_len = B * [1]
    kv_len =  seq_lens.cpu()
    attn_logits2 = decode_paged_attn_golden(q, k_buffer, v_buffer, q_len, kv_len, block_table, sm_scale)
    torch.npu.synchronize()

    print("Max", torch.max(torch.abs(attn_logits1 - attn_logits2)))
    print("Acc", torch.allclose(attn_logits1, attn_logits2, rtol=1e-2, atol=1e-2))

    tensors_to_save = {
        # 输入张量
        "q": q,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "kv_seq_lens": seq_lens,
        "attn_logits": attn_logits1,
        "sm_scale": sm_scale,
        "page_size": page_size,
        "block_table": block_table,
        "B": B,
        "H_Q": H_Q,
        "DV": DV
    }

    save_path = "/home/zrx/paged/test_paged_attention.pt"
    # torch.save(tensors_to_save, save_path)
    # print(f"输入和输出张量已保存到：{save_path}")

def test_mla_decode_attention_kernel(B, S, H_Q, H_KV, DQ, DV, dtype):
    device = get_device()
    torch.manual_seed(2)
    seq_lens = torch.full((B, ), S, device=device, dtype=torch.int32)
    page_size = 128
    max_page_num = (S + page_size - 1) // page_size

    q = torch.randn((B, H_Q, DQ), device=device, dtype=dtype)
    k_nope = torch.randn((max_page_num * B, page_size, H_KV, DV), device=device, dtype=dtype)
    k_rope = torch.randn((max_page_num * B, page_size, H_KV, DQ - DV), device=device, dtype=dtype)
    block_table = torch.arange(0, B * max_page_num, device=device, dtype=torch.int32).reshape(B, max_page_num)
    
    attn_logits1 = torch.empty((B, H_Q, DV), device=device, dtype=dtype)
    sm_scale = 1.0 / (DQ**0.5)

    for i in range(10):
        decode_mla(
            q,
            k_nope,
            k_rope,
            attn_logits1,
            seq_lens,
            sm_scale,
            page_size,
            block_table,
        )
    torch.npu.synchronize()


    q_len = B * [1]
    kv_len =  seq_lens.cpu()
    attn_logits2 = decode_mla_golden(q, k_nope, k_nope, k_rope, q_len, kv_len, block_table, sm_scale)
    torch.npu.synchronize()

    print("Max", torch.max(torch.abs(attn_logits1 - attn_logits2)))
    print("Acc", torch.allclose(attn_logits1, attn_logits2, rtol=1e-2, atol=1e-2))


def decode_paged_attn_golden(
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
            assert q_head_num % num_kv_heads == 0, "q_head_num must be divisible by num_kv_heads"
            k = torch.repeat_interleave(k, q_head_num // num_kv_heads, dim=1)
            v = torch.repeat_interleave(v, q_head_num // num_kv_heads, dim=1)
        qk = torch.einsum("qhd,khd->hqk", q, k).float()
        score = torch.softmax(qk, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", score, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


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

        k_nope = key_cache_nope[block_indices].view(-1, num_kv_heads, qk_nope_dim)[:kv_len]
        k_rope = key_cache_rope[block_indices].view(-1, num_kv_heads, qk_rope_dim)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_dim)[:kv_len]

        if q_head_num != num_kv_heads:
            assert q_head_num % num_kv_heads == 0, "q_head_num must be divisible by num_kv_heads"
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


def test_grouped_decode_attention():
    seq_lens = [4096, 3589, 1314, 128]
    dtypes = [torch.bfloat16, torch.float16]
    
    configs = [
        (1, 32, 1, 576, 512),
        (16, 128, 1, 576, 512),
        (16, 128, 1, 288, 256),
        (16, 64, 8, 128, 128)
    ]

    for dtype in dtypes:
        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                test_grouped_decode_attention_kernel(B, S, H_Q, H_KV, D, D_V, dtype)

    configs = [
        (16, 8, 1, 576, 512),
        (16, 32, 1, 576, 512),
        (16, 64, 1, 576, 512),
        (16, 128, 1, 576, 512),
    ]
    for dtype in dtypes:
        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                test_mla_decode_attention_kernel(B, S, H_Q, H_KV, D, D_V, dtype)
        

if __name__ == '__main__':
    if hasattr(torch, "npu") and torch.npu.is_available():
        # (B, N_CTX, H_Q, H_KV, DQ, DV)
        test_grouped_decode_attention()
