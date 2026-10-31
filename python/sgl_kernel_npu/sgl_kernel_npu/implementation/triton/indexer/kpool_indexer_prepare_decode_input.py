"""Fused decode-phase indexer input preparation (Triton-Ascend).

Fuses the 13 small torch ops issued before ``npu_lightning_indexer`` in the
decode path (weights GEMM + pool_seqlens + pool_block_tables) into 2 Triton
kernels:

  Kernel 1 ``_kpool_indexer_prepare_decode_input_gemm_kernel`` (pure Cube, grid=num_aicore)
      w_bf16 = (x @ W.T) * (n_heads**-0.5 * softmax_scale) -> bf16
      Batch-grouped strided assignment: BLOCK_B batches packed into the M dim
      (avoids inefficient M=1 vector-matrix GEMM).

  Kernel 2 ``_kpool_indexer_prepare_decode_input_vv_kernel`` (pure Vector, grid=num_vectorcore)
      Section A: pool_seqlens = seqlens // POOL_SIZE (per-batch split).
      Section B: pool_block_tables = block_tables[:, ::POOL_SIZE]. Loads
      [N, NUM_BLOCKS] contiguously into UB as 1D tiles, then extracts the
      strided columns via reshape + split (avoids the 2x (Index+Transpose)
      materialization of ``build_pooled_page_table`` + ``.contiguous()``).

``actual_seq_lengths_q = arange(1, N+1)`` stays outside the operator.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.implementation.triton.utils.triton_utils import (
    get_device_properties,
)

# Logical tile-element budgets for the accumulator and the two GEMM operands.
_GEMM_ACC_ELEMENTS = 2048
_GEMM_OPERAND_ELEMENTS = 24576


def _choose_gemm_blocks(n_heads: int) -> tuple[int, int]:
    assert (
        0 < n_heads <= 128 and n_heads & (n_heads - 1) == 0
    ), "n_heads must be a power of two in [1, 128]"
    rows = max(1, _GEMM_ACC_ELEMENTS // n_heads)
    block_b = max(16, min(64, 1 << (rows.bit_length() - 1)))
    columns = _GEMM_OPERAND_ELEMENTS // (block_b + n_heads)
    block_k = min(256, 1 << (columns.bit_length() - 1))
    return block_b, block_k


# Kernel 1: pure Cube GEMM for w_bf16


@triton.jit
def _kpool_indexer_prepare_decode_input_gemm_kernel(
    x_ptr,  # bf16* [N, HIDDEN_SIZE]
    w_ptr,  # fp32* [N_HEADS, HIDDEN_SIZE] (ReplicatedLinear weight)
    w_bf16_ptr,  # bf16* [N, N_HEADS]
    N,  # decode batch size (n_real)
    combined_scale,  # n_heads_pow * softmax_scale, inlined at compile time
    HIDDEN_SIZE: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)

    # Pure Cube operator (tl.dot only), partitioned across num_aicore.
    # Use a grid-stride loop over batch groups; pack BLOCK_B batches into the
    # M dimension to avoid inefficient M=1 vector-matrix GEMMs.
    # The w_tile [N_HEADS, BLOCK_K] is batch-independent; load it once per K
    # iteration and reuse it for every batch in the group.
    NUM_B_BLOCKS = (N + BLOCK_B - 1) // BLOCK_B
    for block_idx in range(pid, NUM_B_BLOCKS, NUM_CORES):
        rows = block_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        rows64 = rows.to(tl.int64)
        mask_b = rows < N

        # a_block: x[rows, k0:k0+BLOCK_K] bf16, with batch and K-tail bounds checks.
        a_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(N, HIDDEN_SIZE),
            strides=(HIDDEN_SIZE, 1),
            offsets=(block_idx * BLOCK_B, 0),
            block_shape=(BLOCK_B, BLOCK_K),
            order=(1, 0),
        )
        # w_block: W[n, k0:k0+BLOCK_K] fp32, shared by all batches; advance only along K.
        w_block = tl.make_block_ptr(
            base=w_ptr,
            shape=(N_HEADS, HIDDEN_SIZE),
            strides=(HIDDEN_SIZE, 1),
            offsets=(0, 0),
            block_shape=(N_HEADS, BLOCK_K),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_B, N_HEADS), dtype=tl.float32)
        for k0 in range(0, HIDDEN_SIZE, BLOCK_K):
            # A tile: x[rows, k0:k0+BLOCK_K] bf16 -> fp32
            a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            # B tile: W -> [BLOCK_K, N_HEADS] fp32
            w_k = tl.load(w_block, boundary_check=(1,), padding_option="zero")
            b = tl.trans(w_k)  # [BLOCK_K, N_HEADS]
            acc = tl.dot(a, b, acc)
            a_block = tl.advance(a_block, (0, BLOCK_K))
            w_block = tl.advance(w_block, (0, BLOCK_K))

        w = acc * combined_scale
        w_bf16 = w.to(tl.bfloat16)
        w_out_ptrs = (
            w_bf16_ptr + rows64[:, None] * N_HEADS + tl.arange(0, N_HEADS)[None, :]
        )
        tl.store(w_out_ptrs, w_bf16, mask=mask_b[:, None])


# Kernel 2: pure Vector for pool_seqlens + pool_block_tables


@triton.jit
def _kpool_indexer_prepare_decode_input_vv_kernel(
    seqlens_ptr,  # int32* [N]
    block_tables_ptr,  # int32* [N, NUM_BLOCKS]
    pool_seqlens_ptr,  # int32* [N]
    pool_block_tables_ptr,  # int32* [N, N_POOL]
    N,  # decode batch size (n_real)
    NUM_BLOCKS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_M_A: tl.constexpr,
    BLOCK_M_B: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    N_POOL: tl.constexpr = NUM_BLOCKS // POOL_SIZE
    P_TOTAL: tl.constexpr = BLOCK_M_B // POOL_SIZE

    # Section A: pool_seqlens
    NUM_SEQ_BLOCKS = (N + BLOCK_M_A - 1) // BLOCK_M_A
    for block_idx in range(pid, NUM_SEQ_BLOCKS, NUM_CORES):
        rows = block_idx * BLOCK_M_A + tl.arange(0, BLOCK_M_A)
        mask = rows < N
        s = tl.load(seqlens_ptr + rows, mask=mask)
        pool_s = s // POOL_SIZE
        tl.store(pool_seqlens_ptr + rows, pool_s, mask=mask)

    # Section B: pool_block_tables
    # Treat [N, NUM_BLOCKS] as one flat row of N*NUM_BLOCKS elements, tiled by
    # BLOCK_M_B. Load each tile contiguously into UB, then use reshape+split to
    # extract ::POOL_SIZE (equivalent to block_tables[:, ::POOL_SIZE]).
    # tl.gather does not support int32 and tl.sum fails for int32 reductions,
    # so use the layout-only tl.split operation for extraction.
    NUM_TILES = (N * NUM_BLOCKS + BLOCK_M_B - 1) // BLOCK_M_B

    for tile_idx in range(pid, NUM_TILES, NUM_CORES):
        # 1) Contiguously load BLOCK_M_B elements as a 1D tile for coalesced access.
        offs_in = tile_idx * BLOCK_M_B + tl.arange(0, BLOCK_M_B)
        mask_in = offs_in < N * NUM_BLOCKS
        block = tl.load(
            block_tables_ptr + offs_in, mask=mask_in, other=0
        )  # [BLOCK_M_B]

        # 2) Extract ::4 with reshape + split (POOL_SIZE=4 is hard-coded;
        #    two splits select the even halves).
        block = tl.reshape(block, [P_TOTAL, 2, 2])
        block, _ = tl.split(block)  # [P_TOTAL, 2], select element 0 of each pair (::2)
        block, _ = tl.split(block)  # [P_TOTAL], apply ::2 again to obtain ::4
        pt = block  # [P_TOTAL]

        # 3) Write the 1D tile back contiguously.
        offs_out = tile_idx * P_TOTAL + tl.arange(0, P_TOTAL)
        mask_out = offs_out < N * N_POOL
        tl.store(pool_block_tables_ptr + offs_out, pt, mask=mask_out)


# Wrapper


def kpool_indexer_prepare_decode_input(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
):
    """Fuse decode indexer input preparation into 2 Triton kernels.

    Args:
        x: [N, HIDDEN_SIZE] bf16, hidden state
        weights_proj_weight: [N_HEADS, HIDDEN_SIZE] fp32, ReplicatedLinear weight
        seqlens_32: [N] int32, per-request real KV length
        block_tables: [N, NUM_BLOCKS] int32, 64-page page table (contiguous)
        pool_size: int, pooling size (fixed 4)
        n_heads_pow: float, n_heads**-0.5
        softmax_scale: float, head_dim**-0.5
    Returns:
        w_bf16: [N, N_HEADS] bf16
        pool_seqlens: [N] int32
        pool_block_tables: [N, N_POOL] int32
    """
    # 1. shape constraints
    N, HIDDEN_SIZE = x.shape
    N_HEADS, W_HIDDEN = weights_proj_weight.shape
    assert (
        HIDDEN_SIZE == W_HIDDEN
    ), f"x.shape[-1]={HIDDEN_SIZE} != weights_proj_weight.shape[-1]={W_HIDDEN}"
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert (
        weights_proj_weight.dtype == torch.float32
    ), f"weights_proj_weight must be fp32, got {weights_proj_weight.dtype}"
    assert (
        seqlens_32.dtype == torch.int32
    ), f"seqlens_32 must be int32, got {seqlens_32.dtype}"
    assert (
        block_tables.dtype == torch.int32
    ), f"block_tables must be int32, got {block_tables.dtype}"
    assert seqlens_32.shape == (N,), f"seqlens_32 shape {seqlens_32.shape} != ({N},)"
    assert (
        block_tables.shape[0] == N
    ), f"block_tables.shape[0]={block_tables.shape[0]} != N={N}"
    NUM_BLOCKS = block_tables.shape[1]
    assert x.is_contiguous(), "x must be contiguous"
    assert weights_proj_weight.is_contiguous(), "weights_proj_weight must be contiguous"
    assert seqlens_32.is_contiguous(), "seqlens_32 must be contiguous"
    assert block_tables.is_contiguous(), "block_tables must be contiguous"
    # The split-based ::pool extraction below needs a pool_size-multiple
    # width. Short decode sequences (e.g. a 1-block warmup request) can leave
    # the table narrower than pool_size; pad with dummy pages — entries past
    # pool_seqlens (= seqlens_32 // pool_size) are never consumed downstream,
    # and padded width == ceil-div of the real width, matching the torch-side
    # build_pooled_page_table slicing.
    padded_blocks = -(-NUM_BLOCKS // pool_size) * pool_size or pool_size
    if padded_blocks != NUM_BLOCKS:
        block_tables = F.pad(block_tables, (0, padded_blocks - NUM_BLOCKS))
    NUM_BLOCKS = padded_blocks
    N_POOL = NUM_BLOCKS // pool_size

    # 2. GEMM tiling depends on head count; Vector tiling is unchanged.
    block_b, block_k = _choose_gemm_blocks(N_HEADS)
    BLOCK_M_A = 128
    BLOCK_M_B = 4096
    # GEMM (Cube) uses num_aicore; VV (pure Vector) uses num_vectorcore.
    num_aicore, num_vectorcore = get_device_properties()

    assert (
        BLOCK_M_B % pool_size == 0
    ), f"BLOCK_M_B={BLOCK_M_B} not divisible by pool_size={pool_size}"
    assert pool_size == 4, f"pool_size={pool_size} must be 4"

    # 3. output tensors
    w_bf16 = torch.empty((N, N_HEADS), dtype=torch.bfloat16, device=x.device)
    pool_seqlens = torch.empty((N,), dtype=torch.int32, device=x.device)
    pool_block_tables = torch.empty((N, N_POOL), dtype=torch.int32, device=x.device)

    # 4. combined_scale (inlined at compile time)
    combined_scale = float(n_heads_pow * softmax_scale)

    # 5. kernel 1 (pure Cube GEMM), grid=num_aicore
    _kpool_indexer_prepare_decode_input_gemm_kernel[(num_aicore,)](
        x,
        weights_proj_weight,
        w_bf16,
        N,
        combined_scale,
        HIDDEN_SIZE=HIDDEN_SIZE,
        N_HEADS=N_HEADS,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        NUM_CORES=num_aicore,
    )

    # 6. kernel 2 (pure Vector: pool_seqlens + pool_block_tables), grid=num_vectorcore
    #    No explicit inter-kernel synchronization is needed (weak HBM
    #    store->load dependency; NPU preserves ordering by default).
    _kpool_indexer_prepare_decode_input_vv_kernel[(num_vectorcore,)](
        seqlens_32,
        block_tables,
        pool_seqlens,
        pool_block_tables,
        N,
        NUM_BLOCKS=NUM_BLOCKS,
        POOL_SIZE=pool_size,
        BLOCK_M_A=BLOCK_M_A,
        BLOCK_M_B=BLOCK_M_B,
        NUM_CORES=num_vectorcore,
    )

    return w_bf16, pool_seqlens, pool_block_tables
