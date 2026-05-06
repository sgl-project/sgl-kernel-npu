import torch

import sgl_kernel_npu  # noqa: F401
from debug_chunk_h import stats_ok, total_chunks


def ref_solve_tril(a, chunk_size, cu_seqlens):
    _, total_tokens, h, _ = a.shape
    out = torch.zeros(1, total_tokens, h, chunk_size, dtype=torch.float32)
    cu = cu_seqlens.cpu().tolist()
    af = a.float()
    for bos, eos in zip(cu, cu[1:]):
        for offset in range(0, eos - bos, chunk_size):
            start = bos + offset
            end = min(start + chunk_size, eos)
            valid = end - start
            for head in range(h):
                tile = af[0, start:end, head, :valid]
                inv = torch.linalg.inv(torch.eye(valid) + tile)
                out[0, start:end, head, :valid] = inv
    return out


def block_dim():
    try:
        return int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))
    except (RuntimeError, AssertionError):
        return 24


def main():
    device = torch.device("npu")
    total_tokens = 256
    chunk_size = 128
    h = 16
    cu = torch.tensor([0, 64, total_tokens], device=device, dtype=torch.int32)

    torch.manual_seed(42)
    t_within = torch.zeros(total_tokens, dtype=torch.long)
    cu_list = cu.cpu().tolist()
    for start, end in zip(cu_list, cu_list[1:]):
        t_within[start:end] = torch.arange(end - start) % chunk_size
    chunk_mask = (t_within[:, None] > torch.arange(chunk_size)[None, :]).float()
    a_cpu = torch.randn(1, total_tokens, h, chunk_size) * 0.1 * chunk_mask[None, :, None, :]
    a = a_cpu.to(torch.float16).to(device)

    out = torch.zeros_like(a, dtype=torch.float32)
    minus_identity = torch.zeros(chunk_size, chunk_size, device=device, dtype=torch.float16)
    minus_identity.fill_diagonal_(-1)
    num_matrices = total_chunks(cu, chunk_size) * h

    torch.ops.npu.tri_inverse_debug(
        out, a, minus_identity, cu, min(block_dim(), num_matrices),
        chunk_size, num_matrices, h, True
    )
    torch.npu.synchronize()

    expected = ref_solve_tril(a.cpu(), chunk_size, cu.cpu())
    actual = out.cpu()
    print("tri_inverse max_abs:", (actual - expected).abs().max().item())
    print("tri_inverse ok:", stats_ok(actual, expected.float()))


if __name__ == "__main__":
    main()
