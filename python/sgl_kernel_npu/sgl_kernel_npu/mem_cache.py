import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

@triton.jit
def write_req_to_token_pool_triton_npu(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    # TODO: Block load by pointer could not be compiled with Triton-Ascend at the moment
    #       Scalar loading should be removed as soon as pointer load will be implemented
    for i in range(0, pre_len):
        value = tl.load(prefix_tensor + i)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + i,
            value,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )

def write_cache_indices_npu(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    prefix_pointers = torch.tensor(
        [t.data_ptr() for t in prefix_tensors],
        device=req_to_token_pool.device,
        dtype=torch.uint64,
    )
    # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
    write_req_to_token_pool_triton_npu[(req_pool_indices_tensor.shape[0],)](
        req_to_token_pool.req_to_token,
        req_pool_indices_tensor,
        prefix_pointers,
        prefix_lens_tensor,
        seq_lens_tensor,
        extend_lens_tensor,
        out_cache_loc,
        req_to_token_pool.req_to_token.shape[1],
    )
