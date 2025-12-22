import torch
import torch_npu
import argparse
import sgl_kernel_npu

def anti_quant_fp8(weight, scale, K, N):
    device = weight.device

    scale_fp8_to_32 = torch.tensor(0x7B80, dtype=torch.uint16, device=device)

    w = weight.to(torch.int16)
    bf16_bits = ((w & 0x0080) << 8) | ((w & 0x007F) << 4)
    bf16_bits_u16 = bf16_bits.to(torch.uint16)

    deq = bf16_bits_u16.view(torch.bfloat16) * scale_fp8_to_32.view(torch.bfloat16)
    deq = deq.to(torch.float32)

    deq = (
        deq.reshape(K // 128, 128, N // 128, 128)
           .permute(0, 2, 1, 3)
           .contiguous()
    )

    deq = deq.reshape(-1, 128 * 128) * scale.reshape(-1).to(torch.float32).unsqueeze(-1)

    return (
        deq.to(torch.bfloat16)
           .reshape(K // 128, N // 128, 128, 128)
           .permute(0, 2, 1, 3)
           .contiguous()
           .reshape(K, N)
    )

def compare_fp8_w8a16(M, N, K):
    atol=1e-2
    rtol=1e-2
    seed=0
    torch.manual_seed(seed)

    scale_K = (K + 127) // 128
    scale_N = (N + 127) // 128

    a_bf16 = torch.full([M, K], 1, dtype=torch.bfloat16)
    b_int8 = torch.randint(0, 255, [K, N], dtype=torch.uint8)
    scales = torch.randn([scale_K, scale_N], dtype=torch.float32)
    b_bf16 = anti_quant_fp8(b_int8, scales, K, N)

    out_golden = (a_bf16.to(torch.float32) @ b_bf16.to(torch.float32)).to(torch.bfloat16)

    torch.npu.synchronize()

    out_op = torch.ops.npu.fp8_w8a16_matmul(a_bf16.to("npu"), b_int8.to("npu"), scales.to("npu"), "bf16")

    torch.npu.synchronize()

    g = out_golden.to(torch.float32).to("cpu")
    o = out_op.to(torch.float32).to("cpu")

    print(out_op)
    print(out_golden)
    diff = (o - g).abs()

    ok = torch.allclose(o, g, atol=atol, rtol=rtol)
    max_abs = diff.max().item()
    max_rel = (diff / (g.abs() + 1e-6)).max().item()

    mean_abs = diff.mean().item()
    mean_rel = (diff / (g.abs() + 1e-6)).mean().item()

    bad = diff > (atol + rtol * g.abs())
    bad_cnt = int(bad.sum().item())
    total = bad.numel()

    print(f"[M,N,K]=[{M},{N},{K}] out_dtype=bf16, seed={seed}")
    print(f"scale shape={tuple(scales.shape)} dtype={scales.dtype} stride={scales.stride()}")
    print(
        f"allclose={ok}  bad={bad_cnt}/{total}  "
        f"max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
        f"max_rel={max_rel:.6g}  mean_rel={mean_rel:.6g}"
    )

    if not ok:
        idx = bad.view(-1).nonzero()
        if idx.numel() > 0:
            i = int(idx[0].item())
            row = i // N
            col = i % N
            print(
                f"first_bad at ({row},{col}): "
                f"op={o[row,col].item():.6g} golden={g[row,col].item():.6g} diff={diff[row,col].item():.6g}"
            )

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    compare_fp8_w8a16(args.m, args.n, args.k)
