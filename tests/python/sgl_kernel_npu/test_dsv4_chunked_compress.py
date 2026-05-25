import argparse
from dataclasses import dataclass

import torch


@dataclass
class CaseConfig:
    name: str
    ratio: int
    prefix_len: int
    chunk_len: int
    head_dim: int
    page_size: int


def make_page_table(total_len: int, page_size: int) -> torch.Tensor:
    num_pages = (total_len + page_size - 1) // page_size + 2
    return torch.arange(num_pages, dtype=torch.int64)


def fill_state_pool(
    kv_full: torch.Tensor,
    score_full: torch.Tensor,
    ape: torch.Tensor,
    prefix_len: int,
    ratio: int,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    coff_d = kv_full.shape[1]
    state = kv_full.new_empty((page_table.numel() * page_size, 2 * coff_d))
    state[:, :coff_d].zero_()
    state[:, coff_d:].fill_(float("-inf"))
    for gp in range(prefix_len):
        slot = int(page_table[gp // page_size]) * page_size + gp % page_size
        state[slot, :coff_d] = kv_full[gp]
        state[slot, coff_d:] = score_full[gp] + ape[gp % ratio]
    return state


def gather_state_window(
    state_pool: torch.Tensor,
    page_table: torch.Tensor,
    state_base: int,
    prefix_len: int,
    page_size: int,
) -> torch.Tensor:
    if prefix_len <= state_base:
        return state_pool.new_empty((0, state_pool.shape[-1]))
    gps = torch.arange(state_base, prefix_len, dtype=torch.long)
    slots = page_table[gps // page_size].to(torch.long) * page_size + (gps % page_size)
    return state_pool[slots].contiguous()


def reference_chunked_compress(
    kv_full: torch.Tensor,
    score_full: torch.Tensor,
    state_kv_score_window: torch.Tensor,
    state_base: int,
    ape: torch.Tensor,
    case: CaseConfig,
) -> torch.Tensor:
    overlap = case.ratio == 4
    coff_d = kv_full.shape[1]
    d = case.head_dim
    prefix_len = case.prefix_len
    chunk_kv = kv_full[prefix_len : prefix_len + case.chunk_len]
    chunk_score = score_full[prefix_len : prefix_len + case.chunk_len]
    first_k = prefix_len // case.ratio
    last_k_exclusive = (prefix_len + case.chunk_len) // case.ratio
    outs = []

    for k in range(first_k, last_k_exclusive):
        kv_rows = []
        score_rows = []
        n_src = 2 * case.ratio if overlap else case.ratio
        for src_i in range(n_src):
            is_prev = overlap and src_i >= case.ratio
            j = src_i - case.ratio if is_prev else src_i
            if is_prev and k == 0:
                kv_row = kv_full.new_zeros((d,))
                score_row = kv_full.new_full((d,), float("-inf"))
                kv_rows.append(kv_row)
                score_rows.append(score_row)
                continue

            gp = (k - 1) * case.ratio + j if is_prev else k * case.ratio + j
            if gp < prefix_len:
                state_row = gp - state_base
                kv_row = state_kv_score_window[state_row, :coff_d]
                score_row = state_kv_score_window[state_row, coff_d:]
            else:
                local_idx = gp - prefix_len
                kv_row = chunk_kv[local_idx]
                score_row = chunk_score[local_idx] + ape[j]

            if overlap:
                if is_prev:
                    kv_row = kv_row[:d]
                    score_row = score_row[:d]
                else:
                    kv_row = kv_row[d:]
                    score_row = score_row[d:]
            kv_rows.append(kv_row)
            score_rows.append(score_row)

        kv_stack = torch.stack(kv_rows, dim=0)
        score_stack = torch.stack(score_rows, dim=0)
        outs.append((kv_stack * score_stack.softmax(dim=0)).sum(dim=0))

    if not outs:
        return kv_full.new_empty((0, d))
    return torch.stack(outs, dim=0)


def run_case(
    case: CaseConfig,
    device: torch.device,
    atol: float,
    rtol: float,
    dtype: torch.dtype = torch.float32,
):
    from sgl_kernel_npu.attention.dsv4_chunked_compress import (
        dsv4_chunked_prefill_compress,
    )

    overlap = case.ratio == 4
    coff = 2 if overlap else 1
    total_len = case.prefix_len + case.chunk_len

    # Reference always runs in fp32 for numerical accuracy.
    kv_full = torch.randn(total_len, coff * case.head_dim, dtype=torch.float32)
    score_full = torch.randn(total_len, coff * case.head_dim, dtype=torch.float32) * 0.3
    ape = torch.randn(case.ratio, coff * case.head_dim, dtype=torch.float32) * 0.5
    page_table = make_page_table(total_len, case.page_size)
    state_pool = fill_state_pool(
        kv_full,
        score_full,
        ape,
        case.prefix_len,
        case.ratio,
        page_table,
        case.page_size,
    )
    state_base = max(
        0,
        (case.prefix_len // case.ratio - (1 if overlap else 0)) * case.ratio,
    )
    state_window = gather_state_window(
        state_pool, page_table, state_base, case.prefix_len, case.page_size
    )

    ref = reference_chunked_compress(
        kv_full, score_full, state_window, state_base, ape, case
    )
    # Kernel runs with the requested dtype; output is cast back to fp32 for comparison.
    out = dsv4_chunked_prefill_compress(
        kv_full[case.prefix_len :].contiguous().to(dtype).to(device),
        score_full[case.prefix_len :].contiguous().to(dtype).to(device),
        state_window.contiguous().to(dtype).to(device),
        ape.contiguous().to(dtype).to(device),
        prefix_len=case.prefix_len,
        chunk_len=case.chunk_len,
        ratio=case.ratio,
        overlap=overlap,
        state_base=state_base,
    )
    torch.npu.synchronize()
    out_cpu = out.cpu().float()
    torch.testing.assert_close(out_cpu, ref, atol=atol, rtol=rtol)
    max_diff = (out_cpu - ref).abs().max().item() if ref.numel() else 0.0
    dtype_tag = "fp32" if dtype == torch.float32 else "bf16"
    print(
        f"[PASS] {case.name}[{dtype_tag}]: shape={tuple(ref.shape)} max_diff={max_diff:.6g}"
    )


def run_equivalence_case(
    name: str,
    ratio: int,
    head_dim: int,
    page_size: int,
    device: torch.device,
    atol: float,
    rtol: float,
    extra_chunks: int = 0,
) -> None:
    """Cross-validate chunked path against the non-chunked reference.

    With extra_chunks=0 (default): total_len=2*ratio, prefix_len=ratio,
    chunk_len=ratio → kernel emits n_out=1, validated against non-chunked k=1.

    With extra_chunks=N: total_len=(2+N)*ratio, prefix_len=ratio,
    chunk_len=(1+N)*ratio → kernel emits n_out=1+N, all outputs validated
    against the non-chunked reference. This exercises the out_i loop and the
    store address arithmetic (out + out_i * HEAD_DIM) for out_i > 0.
    """
    from sgl_kernel_npu.attention.dsv4_chunked_compress import (
        dsv4_chunked_prefill_compress,
    )

    overlap = ratio == 4
    coff = 2 if overlap else 1
    prefix_len = ratio
    chunk_len = (1 + extra_chunks) * ratio
    total_len = prefix_len + chunk_len
    n_out_expected = chunk_len // ratio  # == 1 + extra_chunks

    kv_full = torch.randn(total_len, coff * head_dim, dtype=torch.float32)
    score_full = torch.randn(total_len, coff * head_dim, dtype=torch.float32) * 0.3
    ape = torch.randn(ratio, coff * head_dim, dtype=torch.float32) * 0.5

    # Non-chunked reference: prefix=0, all total_len tokens. Yields outputs k=0..n_out.
    # We compare against outputs k=1..n_out (the part the chunked kernel covers).
    nc_case = CaseConfig(f"{name}_nc", ratio, 0, total_len, head_dim, page_size)
    empty_state = torch.empty((0, 2 * coff * head_dim), dtype=torch.float32)
    ref_all = reference_chunked_compress(
        kv_full, score_full, empty_state, 0, ape, nc_case
    )
    ref_chunked = ref_all[1:]  # shape (n_out_expected, head_dim)

    page_table = make_page_table(total_len, page_size)
    state_pool = fill_state_pool(
        kv_full, score_full, ape, prefix_len, ratio, page_table, page_size
    )
    state_base = max(0, (prefix_len // ratio - (1 if overlap else 0)) * ratio)
    state_window = gather_state_window(
        state_pool, page_table, state_base, prefix_len, page_size
    )
    out = dsv4_chunked_prefill_compress(
        kv_full[prefix_len:].contiguous().to(device),
        score_full[prefix_len:].contiguous().to(device),
        state_window.contiguous().to(device),
        ape.contiguous().to(device),
        prefix_len=prefix_len,
        chunk_len=chunk_len,
        ratio=ratio,
        overlap=overlap,
        state_base=state_base,
    )
    torch.npu.synchronize()
    out_cpu = out.cpu()
    assert out_cpu.shape[0] == n_out_expected, (
        f"expected {n_out_expected} outputs, got {out_cpu.shape[0]}"
    )
    torch.testing.assert_close(out_cpu, ref_chunked, atol=atol, rtol=rtol)
    diff = (out_cpu - ref_chunked).abs().max().item()
    suffix = f"_x{1 + extra_chunks}" if extra_chunks else ""
    print(f"[PASS] equivalence/{name}{suffix}: n_out={n_out_expected} max_diff={diff:.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Import failed: {exc}") from exc

    if not torch.npu.is_available():
        raise SystemExit("NPU device not available")

    torch.manual_seed(args.seed)
    device = torch.device("npu")

    # ── fp32 baseline cases (original) ───────────────────────────────────────
    fp32_cases = [
        CaseConfig("overlap_k0_mixed_state", 4, 2, 6, 16, 4),
        CaseConfig("overlap_unaligned_followup", 4, 6, 6, 16, 4),
        CaseConfig("non_overlap_remainder_complete", 128, 50, 80, 16, 16),
        CaseConfig("non_overlap_boundary_followup", 128, 128, 256, 16, 16),
        # ── Item 9: head_dim=128 exercises the multi-col-block path (BLOCK_D=64,
        #    NUM_COL_BLOCKS=2). This is DSV4's actual attention head_dim.
        CaseConfig("overlap_hd128_k0_mixed_state", 4, 2, 6, 128, 4),
        CaseConfig("overlap_hd128_followup", 4, 6, 6, 128, 4),
        CaseConfig("non_overlap_hd128_boundary", 128, 128, 256, 128, 16),
        # ── Item 9: n_out=0 early-return path
        # prefix_len=128, chunk_len=64, ratio=128 → (128+64)//128 - 128//128 = 1-1 = 0
        CaseConfig("n_out_zero_non_overlap", 128, 128, 64, 16, 16),
        # prefix_len=4, chunk_len=2, ratio=4 → (4+2)//4 - 4//4 = 1-1 = 0
        CaseConfig("n_out_zero_overlap", 4, 4, 2, 16, 4),
        # ── overlap with prefix_len exactly a multiple of ratio and n_out > 0.
        # k=1: prev sources [0,4) all from state, current sources [4,8) all from chunk —
        # a clean state/chunk partition that no other case covers.
        CaseConfig("overlap_aligned_prefix_multi_out", 4, 4, 8, 16, 4),
        CaseConfig("overlap_aligned_prefix_multi_out_hd128", 4, 4, 8, 128, 4),
    ]
    for case in fp32_cases:
        run_case(case, device, args.atol, args.rtol, dtype=torch.float32)

    # ── bf16 cases (Item 2) ───────────────────────────────────────────────────
    # bf16 has ~3 decimal digits of precision; kernel accumulates in fp32
    # internally, so tolerance of 2e-2 covers the round-trip cast error.
    BF16_ATOL, BF16_RTOL = 2e-2, 1e-2
    bf16_cases = [
        CaseConfig("bf16_overlap_k0_mixed_state", 4, 2, 6, 16, 4),
        CaseConfig("bf16_overlap_unaligned_followup", 4, 6, 6, 16, 4),
        CaseConfig("bf16_non_overlap_boundary", 128, 128, 256, 16, 16),
        CaseConfig("bf16_overlap_hd128", 4, 2, 6, 128, 4),
        CaseConfig("bf16_non_overlap_hd128", 128, 128, 256, 128, 16),
    ]
    for case in bf16_cases:
        run_case(case, device, BF16_ATOL, BF16_RTOL, dtype=torch.bfloat16)

    # ── Item 5: chunked vs non-chunked equivalence ────────────────────────────
    # Same input, chunked kernel (prefix_len=ratio) must agree with non-chunked
    # reference (prefix_len=0, full 2*ratio tokens). Catches convention bugs
    # that the self-referential reference_chunked_compress would miss.
    run_equivalence_case("overlap_ratio4_hd16", 4, 16, 4, device, args.atol, args.rtol)
    run_equivalence_case(
        "overlap_ratio4_hd128", 4, 128, 4, device, args.atol, args.rtol
    )
    run_equivalence_case(
        "non_overlap_ratio128_hd16", 128, 16, 16, device, args.atol, args.rtol
    )
    run_equivalence_case(
        "non_overlap_ratio128_hd128", 128, 128, 16, device, args.atol, args.rtol
    )

    # ── Multi-output equivalence: extra_chunks=2 → n_out=3, validates
    # out_i loop and store address (out + out_i * HEAD_DIM) for out_i > 0.
    run_equivalence_case(
        "overlap_ratio4_hd16", 4, 16, 4, device, args.atol, args.rtol, extra_chunks=2
    )
    run_equivalence_case(
        "overlap_ratio4_hd128", 4, 128, 4, device, args.atol, args.rtol, extra_chunks=2
    )
    run_equivalence_case(
        "non_overlap_ratio128_hd16",
        128,
        16,
        16,
        device,
        args.atol,
        args.rtol,
        extra_chunks=2,
    )
    run_equivalence_case(
        "non_overlap_ratio128_hd128",
        128,
        128,
        16,
        device,
        args.atol,
        args.rtol,
        extra_chunks=2,
    )

    print("All DeepSeek-V4 chunked compress Triton tests passed.")


if __name__ == "__main__":
    main()
