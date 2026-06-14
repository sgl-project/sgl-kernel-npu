import sgl_kernel_npu
import torch
import torch_npu

BLOCK_SIZE = 128


def apply_group_block_scale(weight_fp8, scale, group_count, k, n, block_size=BLOCK_SIZE):
    """
    Reference dequant for grouped FP8 tensor + per-group per-(128,128) block scale.
    weight_fp8: torch.float8_e4m3fn, shape [g, K, N]
    scale:      float32, shape [g, ceil(K/128), ceil(N/128)]
    return:     bf16, shape [g, K, N]
    """
    if weight_fp8.shape != (group_count, k, n):
        raise ValueError(
            f"weight shape mismatch: expect {(group_count, k, n)}, got {tuple(weight_fp8.shape)}"
        )
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"weight dtype mismatch: expect torch.float8_e4m3fn, got {weight_fp8.dtype}"
        )

    scale_k = (k + block_size - 1) // block_size
    scale_n = (n + block_size - 1) // block_size
    if scale.shape != (group_count, scale_k, scale_n):
        raise ValueError(
            "scale shape mismatch: expect "
            f"{(group_count, scale_k, scale_n)}, got {tuple(scale.shape)}"
        )

    device = weight_fp8.device
    deq_base = weight_fp8.to(torch.float32)

    k_pad = scale_k * block_size
    n_pad = scale_n * block_size

    deq_padded = torch.zeros(
        (group_count, k_pad, n_pad), dtype=torch.float32, device=device
    )
    deq_padded[:, :k, :n] = deq_base

    blocked = (
        deq_padded.view(group_count, scale_k, block_size, scale_n, block_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    blocked = blocked * scale.to(torch.float32).view(
        group_count, scale_k, scale_n, 1, 1
    )

    deq = (
        blocked.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(group_count, k_pad, n_pad)
    )[:, :k, :n]

    return deq.to(torch.bfloat16)


def build_group_list(group_sizes, device):
    """Build an int64 prefix-sum groupList tensor from explicit group sizes."""
    if not group_sizes:
        raise ValueError("group_sizes must not be empty")
    prefix_sum = []
    running = 0
    for group_m in group_sizes:
        if group_m <= 0:
            raise ValueError(f"group size must be positive, got {group_m}")
        running += group_m
        prefix_sum.append(running)
    return torch.tensor(prefix_sum, dtype=torch.int64, device=device)


def group_matmul(a_bf16, b_bf16, group_sizes):
    """Compute grouped matmul reference output on host using explicit group sizes."""
    ret = []
    start = 0
    for group_idx, group_m in enumerate(group_sizes):
        end = start + group_m
        c = a_bf16[start:end, :].to(torch.float32) @ b_bf16[group_idx].to(torch.float32)
        ret.append(c)
        start = end
    ret = torch.cat(ret, dim=0)
    print(f"ret shape: {ret.shape}, {ret.dtype}")
    return ret


def print_gmm_case_summary(
    group_sizes,
    m,
    n,
    k,
    seed,
    b_fp8,
    b_uint8,
    scales,
    group_list,
    out_op,
    out_golden,
    ok,
    bad_cnt,
    total,
    diff,
    atol,
    rtol,
):
    """Print a compact summary for one grouped-matmul correctness case."""
    g = out_golden.to(torch.float32).cpu()
    o = out_op.to(torch.float32).cpu()

    if m <= 8 and n <= 16:
        print("out_op:")
        print(out_op)
        print("out_golden:")
        print(out_golden)

    max_abs = diff.max().item()
    max_rel = (diff / (g.abs() + 1e-6)).max().item()
    mean_abs = diff.mean().item()
    mean_rel = (diff / (g.abs() + 1e-6)).mean().item()

    print(f"[group_sizes,M,N,K]=[{group_sizes},{m},{n},{k}] out_dtype={out_op.dtype}, seed={seed}")
    print(
        f"weight_fp8 shape={tuple(b_fp8.shape)} dtype={b_fp8.dtype} stride={b_fp8.stride()}"
    )
    print(
        f"weight_bits shape={tuple(b_uint8.shape)} dtype={b_uint8.dtype} stride={b_uint8.stride()}"
    )
    print(f"groupList={group_list.cpu().tolist()}")
    print(
        f"scale shape={tuple(scales.shape)} dtype={scales.dtype} stride={scales.stride()}"
    )
    print(
        f"allclose={ok}  bad={bad_cnt}/{total}  "
        f"max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
        f"max_rel={max_rel:.6g}  mean_rel={mean_rel:.6g}"
    )

    if not ok:
        idx = (diff > (atol + rtol * g.abs())).view(-1).nonzero()
        if idx.numel() > 0:
            i = int(idx[0].item())
            row = i // n
            col = i % n
            print(
                f"first_bad at ({row},{col}): "
                f"op={o[row, col].item():.6g} "
                f"golden={g[row, col].item():.6g} "
                f"diff={diff[row, col].item():.6g}"
            )


def compare_fp8_w8a16_grouped(group_sizes, n, k):
    """Run one grouped softfp8 w8a16 case and compare NPU output to reference."""
    atol = 1e-2
    rtol = 1e-2
    seed = 0
    torch.manual_seed(seed)

    group_count = len(group_sizes)
    m = sum(group_sizes)
    scale_k = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16)
    b_fp32 = torch.randn((group_count, k, n), dtype=torch.float32) * 0.5
    b_fp8 = b_fp32.to(torch.float8_e4m3fn)
    b_uint8 = b_fp8.view(torch.uint8)
    scales = torch.rand((group_count, scale_k, scale_n), dtype=torch.float32) + 1e-3
    group_list = build_group_list(group_sizes, "npu")

    b_bf16 = apply_group_block_scale(b_fp8, scales, group_count, k, n)
    out_golden = group_matmul(a_bf16, b_bf16, group_sizes)

    torch.npu.synchronize()

    out_op = torch.ops.npu.softfp8_w8a16_grouped_matmul(
        a_bf16.to("npu"), b_uint8.to("npu"), scales.to("npu"), group_list, "bf16"
    )

    torch.npu.synchronize()

    g = out_golden.to(torch.float32).cpu()
    o = out_op.to(torch.float32).cpu()

    diff = (o - g).abs()

    ok = torch.allclose(o, g, atol=atol, rtol=rtol)

    bad = diff > (atol + rtol * g.abs())
    bad_cnt = int(bad.sum().item())
    total = bad.numel()
    print_gmm_case_summary(
        group_sizes,
        m,
        n,
        k,
        seed,
        b_fp8,
        b_uint8,
        scales,
        group_list,
        out_op,
        out_golden,
        ok,
        bad_cnt,
        total,
        diff,
        atol,
        rtol,
    )

    return ok


if __name__ == "__main__":
    test_cases = [
        ([1], 128, 128),
        ([1, 2], 257, 385),
        ([1, 1, 2], 127, 127),
        ([2, 2, 1, 3], 255, 256),
        ([3, 5, 8], 511, 777),
        ([7, 9, 15], 146, 129),
        ([10, 12, 11, 21], 654, 191),
        ([11, 22, 42], 733, 385),
        ([13, 20], 795, 446),
        ([17, 23, 29], 877, 447),
        ([32, 31, 48], 913, 703),
        ([256, 512, 256], 4096, 4096),
    ]
    pass_cnt = 0
    fail_cnt = 0
    failed_cases = []
    for idx, (group_sizes, n, k) in enumerate(test_cases, 1):
        print("=" * 100)
        print(
            f"[{idx}/{len(test_cases)}] Running case: "
            f"group_sizes={group_sizes}, N={n}, K={k}"
        )
        ok = compare_fp8_w8a16_grouped(group_sizes, n, k)
        if ok:
            pass_cnt += 1
        else:
            fail_cnt += 1
            failed_cases.append((group_sizes, n, k))
    print("=" * 100)
    print(f"Summary: pass={pass_cnt}, fail={fail_cnt}, total={len(test_cases)}")
    if failed_cases:
        print("Failed cases:")
        for case in failed_cases:
            print(f"  group_sizes,N,K={case}")
