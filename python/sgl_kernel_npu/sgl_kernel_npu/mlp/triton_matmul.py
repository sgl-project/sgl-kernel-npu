import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver


def get_npu_aicore_num():
    """Get the number of AI cores available on the NPU device."""
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


@triton.jit
def _matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N,
    K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Traditional row-wise partitioning
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        # Initialize accumulator block
        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Perform blocked matrix multiplication
        for k_start in range(0, K, BLOCK_K):
            # Load block from matrix A
            mat_a_offset = (
                ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None]
                + (k_start + tl.arange(0, BLOCK_K))[None, :]
            )
            mat_a_mask = (
                ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None]
                & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
            )
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            tl.compile_hint(mat_a_block, "dot_pad_only_k")

            # Load block from matrix B
            mat_b_offset = (
                ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None]
                + (n_start + tl.arange(0, BLOCK_N))[None, :]
            )
            mat_b_mask = (
                ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None]
                & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            )
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            tl.compile_hint(mat_b_block, "dot_pad_only_k")

            # Accumulate the dot product
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)

        # Store result to matrix C
        mat_c_offset = (
            ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None]
            + (n_start + tl.arange(0, BLOCK_N))[None, :]
        )
        mat_c_mask = (
            ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None]
            & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
        )
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)


def triton_matmul(mat_a, mat_b):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)

    num_cores = get_npu_aicore_num()

    _matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores)
    return mat_c