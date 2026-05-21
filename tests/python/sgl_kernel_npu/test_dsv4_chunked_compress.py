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


def run_case(case: CaseConfig, device: torch.device, atol: float, rtol: float):
    from sgl_kernel_npu.attention.dsv4_chunked_compress import (
        dsv4_chunked_prefill_compress,
    )

    overlap = case.ratio == 4
    coff = 2 if overlap else 1
    total_len = case.prefix_len + case.chunk_len

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
    out = dsv4_chunked_prefill_compress(
        kv_full[case.prefix_len :].contiguous().to(device),
        score_full[case.prefix_len :].contiguous().to(device),
        state_window.contiguous().to(device),
        ape.contiguous().to(device),
        prefix_len=case.prefix_len,
        chunk_len=case.chunk_len,
        ratio=case.ratio,
        overlap=overlap,
        state_base=state_base,
    )
    torch.npu.synchronize()
    out_cpu = out.cpu()
    torch.testing.assert_close(out_cpu, ref, atol=atol, rtol=rtol)
    max_diff = (out_cpu - ref).abs().max().item() if ref.numel() else 0.0
    print(f"[PASS] {case.name}: shape={tuple(ref.shape)} max_diff={max_diff:.6g}")


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
    cases = [
        CaseConfig("overlap_k0_mixed_state", 4, 2, 6, 16, 4),
        CaseConfig("overlap_unaligned_followup", 4, 6, 6, 16, 4),
        CaseConfig("non_overlap_remainder_complete", 128, 50, 80, 16, 16),
        CaseConfig("non_overlap_boundary_followup", 128, 128, 256, 16, 16),
    ]
    for case in cases:
        run_case(case, device, args.atol, args.rtol)
    print("All DeepSeek-V4 chunked compress Triton tests passed.")


if __name__ == "__main__":
    main()
