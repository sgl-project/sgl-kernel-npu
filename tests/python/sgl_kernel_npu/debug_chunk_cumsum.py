import torch

import sgl_kernel_npu  # noqa: F401


def ref_cumsum(g, chunk_size, cu_seqlens):
    out = torch.zeros_like(g, dtype=torch.float32)
    cu = cu_seqlens.cpu().tolist()
    for bos, eos in zip(cu, cu[1:]):
        for offset in range(0, eos - bos, chunk_size):
            start = bos + offset
            end = min(start + chunk_size, eos)
            out[:, start:end, :] = g.float()[:, start:end, :].cumsum(dim=1)
    return out


def main():
    device = torch.device("npu")
    total_tokens = 256
    num_heads = 16
    chunk_size = 128
    cu = torch.tensor([0, 64, total_tokens], device=device, dtype=torch.int32)

    torch.manual_seed(0)
    g = torch.randn(1, total_tokens, num_heads, device=device, dtype=torch.float32)
    g_sum = torch.empty_like(g)

    torch.ops.npu.chunk_cumsum_debug(
        g, g_sum, cu, 16, cu.numel() - 1, total_tokens
    )
    torch.npu.synchronize()

    expected = ref_cumsum(g.cpu(), chunk_size, cu.cpu())
    diff = (g_sum.cpu() - expected).abs()
    print("max_abs:", diff.max().item())
    print("ok:", torch.allclose(g_sum.cpu(), expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
