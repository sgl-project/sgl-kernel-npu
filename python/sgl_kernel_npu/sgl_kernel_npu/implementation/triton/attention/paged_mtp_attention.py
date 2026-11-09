# SPDX-License-Identifier: Apache-2.0
"""Device-length paged GQA for static DSpark draft blocks.

Adapted from shengzhaotian/sglang@d1bf05bd30. The public operator validates
metadata before dispatching here; this module allocates and launches kernels.

``seq_lens`` contains the valid KV end, including the current query block.
``max_kv_len`` is only a scheduling bound; it never controls valid KV reads.
The query block and GQA heads share a matrix tile. Long contexts can split
the KV scan across cores and merge the partial softmax results.
"""

import torch
import triton
import triton.language as tl

from ....utils.triton_utils import get_device_properties


@triton.jit
def _paged_mtp_kernel(
    Q,
    K,
    V,
    O,
    SeqLens,
    CuQ,
    Pages,
    Acc,
    Lse,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kt,
    stride_vp,
    stride_vt,
    stride_ot,
    stride_oh,
    stride_od,
    stride_pb,
    stride_pp,
    scale,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    PACKED_QUERY: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV * SPLITS, tl.num_programs(0)):
        split = task % SPLITS
        kv_head = (task // SPLITS) % HKV
        batch = task // (SPLITS * HKV)
        kv_len = tl.load(SeqLens + batch)
        q_start = tl.load(CuQ + batch)
        chunk = tl.cdiv(tl.cdiv(kv_len, SPLITS), PAGE) * PAGE
        kv_start = split * chunk
        kv_end = tl.minimum(kv_start + chunk, kv_len)

        rows = tl.arange(0, Q_TILE)
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        dims = tl.arange(0, BLOCK_D)
        # Row bounds stay quotient-based for the Q load and score mask: on A5
        # Triton 3.2.0 the equivalent flat rows < GAMMA * GROUP form
        # miscompiles when the tile is exactly full, producing NaNs. The output
        # stores keep the flat form to avoid tail writes past the request.
        # Preserve A5 query addressing; packed Q only changes the input layout.
        if PACKED_QUERY:
            q = tl.load(
                Q
                + (batch * HKV + kv_head) * GAMMA * GROUP * DIM
                + rows[:, None] * DIM
                + dims[None, :],
                mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
                other=0.0,
            )
        else:
            q = tl.load(
                Q
                + (q_start + q_index[:, None]) * stride_qt
                + q_head[:, None] * stride_qh
                + dims[None, :] * stride_qd,
                mask=(q_index[:, None] < GAMMA) & (dims[None, :] < DIM),
                other=0.0,
            )
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        acc = tl.zeros((Q_TILE, BLOCK_D), tl.float32)

        for block in range(tl.cdiv(tl.maximum(kv_end - kv_start, 0), KV_BLOCK)):
            start = kv_start + block * KV_BLOCK
            count = tl.minimum(KV_BLOCK, kv_end - start)
            physical_page = tl.load(
                Pages + batch * stride_pb + (start // PAGE) * stride_pp
            )
            in_page = start % PAGE
            k_ptr = tl.make_block_ptr(
                base=K
                + physical_page * stride_kp
                + kv_head * DIM
                + in_page * stride_kt,
                shape=(DIM, count),
                strides=(1, stride_kt),
                offsets=(0, 0),
                block_shape=(BLOCK_D, BLOCK_N),
                order=(0, 1),
            )
            v_ptr = tl.make_block_ptr(
                base=V
                + physical_page * stride_vp
                + kv_head * DIM
                + in_page * stride_vt,
                shape=(count, DIM),
                strides=(stride_vt, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            keys = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
            values = tl.load(v_ptr, boundary_check=(0, 1), padding_option="zero")
            scores = tl.dot(q, keys) * scale
            kv_positions = start + tl.arange(0, BLOCK_N)
            if KV_BLOCK < BLOCK_N:
                allowed = (tl.arange(0, BLOCK_N)[None, :] < count) & (
                    q_index[:, None] < GAMMA
                )
            else:
                allowed = (kv_positions[None, :] < kv_end) & (q_index[:, None] < GAMMA)
            if CAUSAL:
                allowed = allowed & (
                    kv_positions[None, :] <= kv_len - GAMMA + q_index[:, None]
                )
            scores = tl.where(allowed, scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # A split can be empty for a causal query or a padded request.
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            probability = tl.exp(scores - safe_max[:, None])
            correction = tl.exp(maximum - safe_max)
            denominator = denominator * correction + tl.sum(probability, 1)
            acc = acc * correction[:, None] + tl.dot(probability.to(keys.dtype), values)
            maximum = next_max

        safe_denominator = tl.where(denominator > 0, denominator, 1.0)
        acc = acc / safe_denominator[:, None]
        if SPLITS == 1:
            tl.store(
                O
                + (q_start + q_index[:, None]) * stride_ot
                + q_head[:, None] * stride_oh
                + dims[None, :] * stride_od,
                acc,
                mask=(rows[:, None] < GAMMA * GROUP) & (dims[None, :] < DIM),
            )
        else:
            tl.store(
                Acc + task * Q_TILE * DIM + rows[:, None] * DIM + dims[None, :],
                acc,
                mask=dims[None, :] < DIM,
            )
            lse = tl.where(
                denominator > 0, maximum + tl.log(safe_denominator), float("-inf")
            )
            tl.store(Lse + task * Q_TILE + rows, lse)


@triton.jit
def _reduce_mtp_kernel(
    Acc,
    Lse,
    O,
    CuQ,
    stride_ot,
    stride_oh,
    stride_od,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    DIM: tl.constexpr,
    GAMMA: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    GROUP: tl.constexpr = HQ // HKV
    for task in range(tl.program_id(0), BATCH * HKV, tl.num_programs(0)):
        batch = task // HKV
        kv_head = task % HKV
        q_start = tl.load(CuQ + batch)
        rows = tl.arange(0, Q_TILE)
        dims = tl.arange(0, BLOCK_D)
        maximum = tl.full((Q_TILE,), float("-inf"), tl.float32)
        for split in tl.static_range(SPLITS):
            lse = tl.load(Lse + (task * SPLITS + split) * Q_TILE + rows)
            maximum = tl.maximum(maximum, lse)
        safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
        denominator = tl.zeros((Q_TILE,), tl.float32)
        result = tl.zeros((Q_TILE, BLOCK_D), tl.float32)
        for split in tl.static_range(SPLITS):
            workspace_task = task * SPLITS + split
            lse = tl.load(Lse + workspace_task * Q_TILE + rows)
            weight = tl.exp(lse - safe_max)
            partial = tl.load(
                Acc
                + workspace_task * Q_TILE * DIM
                + rows[:, None] * DIM
                + dims[None, :],
                mask=dims[None, :] < DIM,
                other=0.0,
            )
            result += weight[:, None] * partial
            denominator += weight
        result /= tl.where(denominator > 0, denominator, 1.0)[:, None]
        q_index = rows // GROUP
        q_head = kv_head * GROUP + rows % GROUP
        tl.store(
            O
            + (q_start + q_index[:, None]) * stride_ot
            + q_head[:, None] * stride_oh
            + dims[None, :] * stride_od,
            result,
            mask=(rows[:, None] < GAMMA * GROUP) & (dims[None, :] < DIM),
        )


def paged_mtp_attention_decode(
    *,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gamma: int,
    softmax_scale: float,
    max_kv_len: int,
    is_arch22: bool,
    causal: bool = False,
) -> torch.Tensor:
    """Attend to actual device KV lengths; ``max_kv_len`` only sizes the work.

    Inputs use FIA's ND paged layout ``[pages, page_size, kv_heads * dim]``.
    Zero lengths denote padding, whose output is zero. All request rows have
    ``gamma`` queries; ragged verification and sliding windows are unsupported.
    Causal queries are right aligned: query i sees KV positions through
    ``seq_lens[b] - gamma + i``. All devices share the grouped matrix scan;
    ``is_arch22`` selects A2/A3 launch settings and packs multi-query GQA
    into an affine Q matrix first. The operator layer detects the hardware.
    """
    batch_size = seq_lens.numel()
    output = torch.empty_like(query)
    if not batch_size:
        return output
    cube_cores, vector_cores = get_device_properties()
    group_size = num_q_heads // num_kv_heads
    # A2/A3 use an affine matrix source for the multi-query GQA tile.
    # Packing is captured in NPU graphs; arch35 (A5) keeps its input layout.
    packed_query = is_arch22 and gamma > 1 and group_size > 1
    query_strides = query.stride()
    if packed_query:
        query = (
            query.reshape(batch_size, gamma, num_kv_heads, group_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    tasks = batch_size * num_kv_heads
    splits = 1
    if tasks < cube_cores and max_kv_len >= (256 if group_size == 1 else 2048):
        splits = max(1, min(cube_cores // tasks, max_kv_len // 256))
    q_tile = max(
        32 if is_arch22 else 16,
        triton.next_power_of_2(gamma * group_size),
    )
    block_d = (
        max(32, triton.next_power_of_2(head_dim))
        if is_arch22
        else triton.next_power_of_2(head_dim)
    )
    kv_block = min(128, page_size)
    if is_arch22 and q_tile * block_d >= 8192:
        # Bound the wide-head tile's arch22 local-memory footprint.
        kv_block = min(kv_block, 64)
    # Arch22 uses at least 32 matrix rows/columns for small-head transfers.
    # KV_BLOCK still advances by the physical page width for pages below 32.
    block_n = max(32, kv_block) if is_arch22 else kv_block
    if splits > 1:
        acc = torch.empty(
            (tasks * splits, q_tile, head_dim), dtype=torch.float32, device=query.device
        )
        lse = torch.empty(
            (tasks * splits, q_tile), dtype=torch.float32, device=query.device
        )
    else:
        acc = lse = output  # Unused pointer arguments in the unsplit specialization.
    _paged_mtp_kernel[(cube_cores,)](
        query,
        k_cache,
        v_cache,
        output,
        seq_lens,
        cu_q_lens,
        block_tables,
        acc,
        lse,
        *query_strides,
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        *output.stride(),
        *block_tables.stride(),
        softmax_scale,
        BATCH=batch_size,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        DIM=head_dim,
        PAGE=page_size,
        GAMMA=gamma,
        SPLITS=splits,
        Q_TILE=q_tile,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CAUSAL=causal,
        PACKED_QUERY=packed_query,
        KV_BLOCK=kv_block,
    )
    if splits > 1:
        _reduce_mtp_kernel[(vector_cores,)](
            acc,
            lse,
            output,
            cu_q_lens,
            *output.stride(),
            BATCH=batch_size,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            DIM=head_dim,
            GAMMA=gamma,
            SPLITS=splits,
            Q_TILE=q_tile,
            BLOCK_D=block_d,
        )
    return output
