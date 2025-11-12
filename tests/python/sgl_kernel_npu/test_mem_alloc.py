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
    hidden_size = 6144
    input = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    residual = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    weight = torch.randn(hidden_size).to(torch.bfloat16).npu()
    bias = torch.randn(hidden_size).to(torch.bfloat16).npu()
    res1, res2 = write_cache_indices(input, residual, weight, bias, 1e-6)
    ans1, ans2 = write_cache_indices_golden(input, residual, weight, bias, 1e-6)

    torch.testing.assert_close(res1, ans1, rtol=5e-3)
    torch.testing.assert_close(res2, ans2, rtol=5e-3)

if __name__ == "__main__":
    test_write_cache_indices()
