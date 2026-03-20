import sgl_kernel_npu
import torch
import torch_npu

BLOCK_SIZE = 128


def apply_block_scale(weight_fp8, scale, K, N, block_size=BLOCK_SIZE):
    """
    Reference dequant for valid fp8 tensor + per-(128,128) block scale.
    weight_fp8: torch.float8_e4m3fn, shape [K, N]
    scale:      float32, shape [ceil(K/128), ceil(N/128)]
    return:     bf16, shape [K, N]
    """
    if weight_fp8.shape != (K, N):
        raise ValueError(
            f"weight shape mismatch: expect {(K, N)}, got {tuple(weight_fp8.shape)}"
        )
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"weight dtype mismatch: expect torch.float8_e4m3fn, got {weight_fp8.dtype}"
        )

    scale_K = (K + block_size - 1) // block_size
    scale_N = (N + block_size - 1) // block_size
    if scale.shape != (scale_K, scale_N):
        raise ValueError(
            f"scale shape mismatch: expect {(scale_K, scale_N)}, got {tuple(scale.shape)}"
        )

    device = weight_fp8.device
    deq_base = weight_fp8.to(torch.float32)  # [K, N]

    K_pad = scale_K * block_size
    N_pad = scale_N * block_size

    deq_padded = torch.zeros((K_pad, N_pad), dtype=torch.float32, device=device)
    deq_padded[:K, :N] = deq_base

    blocked = (
        deq_padded.view(scale_K, block_size, scale_N, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    blocked = blocked * scale.to(torch.float32).view(scale_K, scale_N, 1, 1)

    deq = (blocked.permute(0, 2, 1, 3).contiguous().view(K_pad, N_pad))[:K, :N]


def compare_fp8_w8a16(M, N, K):
    atol = 1e-2
    rtol = 1e-2
    seed = 0
    torch.manual_seed(seed)

    scale_K = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_N = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    a_bf16 = torch.full([M, K], 1, dtype=torch.bfloat16)
    b_fp32 = torch.randn([K, N], dtype=torch.float32) * 0.5
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)
    b_int8 = b_fp8.view(torch.uint8)

    scales = torch.rand([scale_K, scale_N], dtype=torch.float32) + 1e-3

    b_bf16 = apply_block_scale(b_fp8, scales, K, N)

    out_golden = (a_bf16.to(torch.float32) @ b_bf16.to(torch.float32)).to(
        torch.bfloat16
    )

    torch.npu.synchronize()

    out_op = torch.ops.npu.softfp8_w8a16_matmul(
        a_bf16.to("npu"), b_int8.to("npu"), scales.to("npu"), "bf16"
    )

    torch.npu.synchronize()

    g = out_golden.to(torch.float32).cpu()
    o = out_op.to(torch.float32).cpu()

    if M <= 8 and N <= 16:
        print("out_op:")
        print(out_op)
        print("out_golden:")
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
    print(
        f"weight_fp8 shape={tuple(b_fp8.shape)} dtype={b_fp8.dtype} stride={b_fp8.stride()}"
    )
    print(
        f"weight_bits shape={tuple(b_int8.shape)} dtype={b_int8.dtype} stride={b_int8.stride()}"
    )
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
            row = i // N
            col = i % N
            print(
                f"first_bad at ({row},{col}): "
                f"op={o[row, col].item():.6g} "
                f"golden={g[row, col].item():.6g} "
                f"diff={diff[row, col].item():.6g}"
            )

    return ok


if __name__ == "__main__":
    test_cases = [
        (1, 128, 128),
        (3, 257, 385),
        (4, 127, 127),
        (8, 255, 256),
        (16, 511, 777),
        (31, 146, 129),
        (46, 435, 190),
        (54, 654, 191),
        (75, 733, 385),
        (33, 795, 446),
        (69, 877, 447),
        (111, 913, 703),
        (1, 4096, 4096),
        (4, 7168, 7168),
        (16, 14336, 4096),
        (32, 28672, 7168),
        (128, 4096, 4096),
        (256, 6144, 6144),
        (1024, 2112, 7168),
        (4096, 6144, 2112),
    ]
    pass_cnt = 0
    fail_cnt = 0
    failed_cases = []
    for idx, (m, n, k) in enumerate(test_cases, 1):
        print("=" * 100)
        print(f"[{idx}/{len(test_cases)}] Running case: M={m}, N={n}, K={k}")
        ok = compare_fp8_w8a16(m, n, k)
        if ok:
            pass_cnt += 1
        else:
            fail_cnt += 1
            failed_cases.append((m, n, k))
    print("=" * 100)
    print(f"Summary: pass={pass_cnt}, fail={fail_cnt}, total={len(test_cases)}")
    if failed_cases:
        print("Failed cases:")
        for case in failed_cases:
            print(f"  M,N,K={case}")
