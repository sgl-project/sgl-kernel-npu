import torch
from sgl_kernel_npu.mem_cache import write_cache_indices

def write_cache_indices_golden(
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_tensor: torch.Tensor,
):
        pt = 0
        for i in range(req_pool_indices.shape[0]):
            req_idx = req_pool_indices[i].item()
            prefix_len = prefix_lens[i].item()
            seq_len = seq_lens[i].item()
            extend_len = extend_lens[i].item()

            req_to_token_tensor[(req_idx, slice(0, prefix_len))] = prefix_tensors[i]
            req_to_token_tensor[(req_idx, slice(prefix_len, seq_len))] = out_cache_loc[pt : pt + extend_len]

            pt += extend_len

def test_write_cache_indices():
    batch_size = 16
    pool_size = 64
    content_len = 256
    page_size = 2
    req_to_token_tensor = torch.zeros((pool_size, context_len), dtype=torch.int32, device="npu")
    req_to_token_tensor_ref = torch.zeros((pool_size, content_len), dtype=torch.int32, device="npu")
    req_pool_indices = torch.tensor(torch.arange(page_size*batch_size, device=self.device).reshape(-1), dtype=torch.int64, device = "npu")

    prefix_lens = torch.randint(16, (batch_size), dtype=torch.int64, device = "npu")
    extend_lens = torch.randint(16, (batch_size), dtype=torch.int64, device = "npu")
    seq_lens = prefix_lens + extended_lens

    prefix_tensors = [torch.randint(256, (batch_size), dtype=torch.int64, device = "npu") for r in range(batch_size)]
    out_cache_loc = torch.arange(page_size*sum(extend_lens), device="npu").reshape(-1)

    write_cache_indices_golden(out_cache_loc, req_pool_indices, prefix_lens, seq_lens, extend_lens, prefix_tensors, req_to_token_tensor_ref)
    write_cache_indices(out_cache_loc, req_pool_indices, prefix_lens, seq_lens, extend_lens, prefix_tensors, req_to_token_tensor)
    torch.testing.assert_close(req_to_token_tensor, req_to_token_tensor_ref, rtol=5e-3)

if __name__ == "__main__":
    test_write_cache_indices()
