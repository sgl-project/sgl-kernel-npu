import argparse

import sgl_kernel_npu
import torch
import torch_npu


def anti_quant_fp8_to_bf16(weight, scale, G, K, N):
    device = weight.device

    # fp8 byte -> bf16 bits trick (keep your original logic)
    scale_fp8_to_32 = torch.tensor(0x7B80, dtype=torch.uint16, device=device)

    w = weight.to(torch.int16)
    bf16_bits = ((w & 0x0080) << 8) | ((w & 0x007F) << 4)  # int16
    bf16_bits_u16 = bf16_bits.to(torch.uint16)

    deq = bf16_bits_u16.view(torch.bfloat16) * scale_fp8_to_32.view(torch.bfloat16)
    deq = deq.to(torch.float32)

    # [K,N] -> blocks [K/128, N/128, 128, 128]
    deq = (
        deq.reshape(G, K // 128, 128, N // 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    )

    # apply per-128x128 scale: flatten blocks -> multiply -> reshape back
    deq = deq.reshape(G, -1, 128 * 128) * scale.reshape(G, -1).to(
        torch.float32
    ).unsqueeze(-1)

    return (
        deq.to(torch.bfloat16)
        .reshape(G, K // 128, N // 128, 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(weight.shape)
    )


def gen_data_fp8(g, row, col):
    data = torch.randn((g, row, col), dtype=torch.float32)
    data_e4m3 = data.to(torch.float8_e4m3fn)
    return data_e4m3


def batch_matmul(a_fp16, b_fp16, g):
    ret = []
    for i in range(g):
        c = a_fp16[i, :, :].to(torch.float32) @ b_fp16[i, :, :].to(torch.float32)
        ret.append(c)
    ret = torch.stack(ret, dim=0)
    print(f"ret shape: {ret.shape}, {ret.dtype}")
    return ret


def compare_data_Wf8Abf16(g, m, n, k, trans_a, trans_b):
    a_bf16 = torch.randn((g, m, k), dtype=torch.bfloat16, device="npu")

    b_fp8 = gen_data_fp8(g, k, n)
    b_int8 = b_fp8.view(torch.int8).to("npu")

    scale_K = (k + 127) // 128
    scale_N = (n + 127) // 128
    scales = torch.randn([g, scale_K, scale_N], dtype=torch.float32, device="npu")

    result = torch.ops.npu.fp8_w8a16_batch_matmul(a_bf16, b_int8, scales, "bf16")

    torch.npu.synchronize()
    b_bf16 = anti_quant_fp8_to_bf16(b_int8, scales, g, k, n)
    c_fp32 = batch_matmul(a_bf16, b_bf16, g)

    print(f"result: {result.shape}, {result.dtype}, {torch.sum(result)}")
    print(f"gt: {c_fp32.shape}, {torch.sum(c_fp32)}")

    gt = c_fp32.to(torch.float32).to("cpu").reshape(-1, n)
    o = result.to(torch.float32).to("cpu").reshape(-1, n)
    diff = (o - gt).abs()

    atol = rtol = 1e-2
    ok = torch.allclose(o, gt, atol=atol, rtol=rtol)
    max_abs = diff.max().item()
    max_rel = (diff / (gt.abs() + 1e-6)).max().item()

    mean_abs = diff.mean().item()
    mean_rel = (diff / (gt.abs() + 1e-6)).mean().item()

    bad = diff > (atol + rtol * gt.abs())
    bad_cnt = int(bad.sum().item())
    total = bad.numel()

    print(f"[B,M,N,K]=[{g},{m},{n},{k}] out_dtype={result.dtype}")
    print(
        f"scale shape={tuple(scales.shape)} dtype={scales.dtype} stride={scales.stride()}"
    )
    print(
        f"allclose={ok}  bad={bad_cnt}/{total}  "
        f"max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
        f"max_rel={max_rel:.6g}  mean_rel={mean_rel:.6g}"
    )

    if not ok:
        idx = bad.view(-1).nonzero()
        if idx.numel() > 0:
            i = int(idx[0].item())
            row = i // n
            col = i % n
            print(
                f"first_bad at ({row},{col}): "
                f"op={o[row,col].item():.6g} golden={gt[row,col].item():.6g} diff={diff[row,col].item():.6g}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 2 256 256 256 0 0
    parser.add_argument("b", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("trans_a", type=int)
    parser.add_argument("trans_b", type=int)
    args = parser.parse_args()
    torch.manual_seed(100)
    compare_data_Wf8Abf16(args.b, args.m, args.n, args.k, args.trans_a, args.trans_b)
