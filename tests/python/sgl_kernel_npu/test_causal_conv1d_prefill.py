"""Tests for the fused causal_conv1d op in prefill mode (run_mode=0)."""

import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


def make_query_start_loc(lengths: Iterable[int], device: torch.device) -> torch.Tensor:
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + int(length))
    if device.type == "cpu":
        return torch.tensor(qsl, device="cpu", dtype=torch.int32)
    out = torch.empty((len(qsl),), device=device, dtype=torch.int32)
    for idx, value in enumerate(qsl):
        out[idx] = int(value)
    return out


def make_device_bool_tensor(values: Iterable[bool], device: torch.device) -> torch.Tensor:
    values = list(values)
    out = torch.zeros((len(values),), device=device, dtype=torch.bool)
    for idx, value in enumerate(values):
        out[idx] = bool(value)
    return out


def make_device_int_tensor(values: Iterable[int], device: torch.device) -> torch.Tensor:
    values = list(values)
    if device.type == "cpu":
        return torch.tensor(values, device="cpu", dtype=torch.int32)
    out = torch.empty((len(values),), device=device, dtype=torch.int32)
    for idx, value in enumerate(values):
        out[idx] = int(value)
    return out


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x


def summarize_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return diff.max().item(), diff.mean().item()


def expect_failure(name: str, fn, expected_substrings: tuple[str, ...]):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if not any(substr in message for substr in expected_substrings):
            raise AssertionError(
                f"{name} failed with unexpected message: {message}"
            ) from exc
        print(f"[PASS] {name}: {message.splitlines()[0]}")
        return
    raise AssertionError(f"{name} unexpectedly succeeded")


# CPU reference: prefill honors has_initial_state and leaves the state tail unchanged.


def reference_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    activation_mode: bool,
    pad_slot_id: int,
):
    width = weight.shape[0]
    state_prefix = width - 1
    dim = x.shape[-1]
    x_tokens = flatten_tokens(x)
    batch = x.shape[0] if x.dim() == 3 else query_start_loc.numel() - 1
    seq_len = x.shape[1] if x.dim() == 3 else None

    y_ref = torch.zeros((x_tokens.shape[0], dim), device=x.device, dtype=torch.float32)
    valid_mask = torch.zeros((x_tokens.shape[0],), device="cpu", dtype=torch.bool)
    conv_states_ref = conv_states.clone()

    weight_fp32 = weight.float()
    bias_fp32 = bias.float() if bias is not None else None

    for seq in range(batch):
        if x.dim() == 3:
            start = seq * seq_len
            length = seq_len
        else:
            start = int(query_start_loc[seq].item())
            end = int(query_start_loc[seq + 1].item())
            length = end - start

        if length <= 0:
            continue

        cache_idx = seq if cache_indices is None else int(cache_indices[seq].item())
        if cache_idx == pad_slot_id:
            continue

        valid_mask[start : start + length] = True

        has_init = (
            bool(has_initial_state[seq].item())
            if has_initial_state is not None
            else False
        )
        if has_init:
            hist = conv_states_ref[cache_idx, :state_prefix].clone()
        else:
            hist = torch.zeros((state_prefix, dim), device=x.device, dtype=x.dtype)

        x_seg = x_tokens[start : start + length]
        ext_raw = torch.cat([hist, x_seg], dim=0)
        ext = ext_raw.float()

        acc = sum(ext[j : j + length] * weight_fp32[j] for j in range(width))
        if bias_fp32 is not None:
            acc = acc + bias_fp32
        if activation_mode:
            acc = F.silu(acc)

        y_ref[start : start + length] = acc.to(x.dtype).float()
        conv_states_ref[cache_idx, :state_prefix] = ext_raw[-state_prefix:]

    return y_ref, conv_states_ref, valid_mask


@dataclass
class PrefillCaseConfig:
    name: str
    dtype: torch.dtype
    dim: int
    width: int
    state_len: int
    num_cache_lines: int
    activation_mode: bool
    use_bias: bool
    input_mode: str  # "3d" or "2d"
    batch: int
    seq_len: Optional[int] = None
    lengths: Optional[list[int]] = None
    cache_indices: Optional[list[int]] = None
    has_initial_state: Optional[list[bool]] = None


def run_prefill_positive_case(
    case: PrefillCaseConfig, device: torch.device, atol: float, rtol: float, pad_slot_id: int
):
    host_device = torch.device("cpu")
    width = case.width
    state_prefix = width - 1

    if case.input_mode == "3d":
        assert case.seq_len is not None
        x_cpu = torch.randn(
            (case.batch, case.seq_len, case.dim), device=host_device, dtype=case.dtype
        )
        lengths = [case.seq_len] * case.batch
    else:
        assert case.lengths is not None
        x_cpu = torch.randn(
            (sum(case.lengths), case.dim), device=host_device, dtype=case.dtype
        )
        lengths = case.lengths

    weight_cpu = torch.randn((case.width, case.dim), device=host_device, dtype=case.dtype)
    bias_cpu = (
        torch.randn((case.dim,), device=host_device, dtype=case.dtype)
        if case.use_bias
        else None
    )
    conv_states_cpu = torch.randn(
        (case.num_cache_lines, case.state_len, case.dim),
        device=host_device,
        dtype=case.dtype,
    )
    cache_indices_cpu = make_device_int_tensor(case.cache_indices, host_device)
    has_initial_state_cpu = make_device_bool_tensor(case.has_initial_state, host_device)
    query_start_loc_cpu = make_query_start_loc(lengths, host_device) if case.input_mode == "2d" else None

    assert len(case.cache_indices) == case.batch
    assert len(case.has_initial_state) == case.batch
    assert (cache_indices_cpu == pad_slot_id).sum().item() < case.batch

    y_ref, conv_states_ref, valid_mask = reference_causal_conv1d_prefill(
        x=x_cpu,
        weight=weight_cpu,
        conv_states=conv_states_cpu,
        query_start_loc=query_start_loc_cpu,
        cache_indices=cache_indices_cpu,
        has_initial_state=has_initial_state_cpu,
        bias=bias_cpu,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
    )

    x = x_cpu.to(device=device)
    weight = weight_cpu.to(device=device)
    bias = bias_cpu.to(device=device) if bias_cpu is not None else None
    conv_states_npu = conv_states_cpu.to(device=device)
    cache_indices = make_device_int_tensor(case.cache_indices, device)
    has_initial_state = make_device_bool_tensor(case.has_initial_state, device)
    query_start_loc = (
        make_query_start_loc(lengths, device) if case.input_mode == "2d" else None
    )

    y_npu = torch.ops.npu.causal_conv1d(
        x=x,
        weight=weight,
        conv_states=conv_states_npu,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
        run_mode=0,
    )
    torch.npu.synchronize()

    y_npu_cpu = flatten_tokens(y_npu).cpu().float()
    y_ref_valid = y_ref[valid_mask]
    y_npu_valid = y_npu_cpu[valid_mask]
    if y_ref_valid.numel() > 0:
        torch.testing.assert_close(y_npu_valid, y_ref_valid, atol=atol, rtol=rtol)

    # State writeback is a plain same-dtype DataCopy -> require bit-exact match.
    torch.testing.assert_close(
        conv_states_npu.cpu().float(),
        conv_states_ref.float(),
        atol=0.0,
        rtol=0.0,
    )

    referenced_rows = {
        int(ci)
        for ci in case.cache_indices
        if ci != pad_slot_id and 0 <= ci < case.num_cache_lines
    }
    untouched_rows = sorted(set(range(case.num_cache_lines)) - referenced_rows)
    if untouched_rows:
        torch.testing.assert_close(
            conv_states_npu[untouched_rows].cpu().float(),
            conv_states_cpu[untouched_rows].float(),
            atol=0.0,
            rtol=0.0,
        )

    out_max, out_mean = (
        summarize_diff(y_npu_valid, y_ref_valid) if y_ref_valid.numel() > 0 else (0.0, 0.0)
    )
    print(
        f"[PASS] {case.name}: prefill "
        f"output(max={out_max:.6g}, mean={out_mean:.6g})"
    )


def run_prefill_negative_cases(device: torch.device, dtype: torch.dtype, pad_slot_id: int):
    dim = 4096
    x = torch.randn((2, 4, dim), device=device, dtype=dtype)
    weight = torch.randn((4, dim), device=device, dtype=dtype)
    conv_states = torch.randn((4, 3, dim), device=device, dtype=dtype)
    cache_indices = make_device_int_tensor([0, 2], device)
    has_initial_state = make_device_bool_tensor([True, False], device)
    bias = torch.randn((dim,), device=device, dtype=dtype)

    def call(xv=x, wv=weight, csv=conv_states, bv=bias, qslv=None, civ=cache_indices, hisv=has_initial_state):
        return torch.ops.npu.causal_conv1d(
            x=xv,
            weight=wv,
            conv_states=csv,
            bias=bv,
            query_start_loc=qslv,
            cache_indices=civ,
            has_initial_state=hisv,
            activation_mode=False,
            pad_slot_id=pad_slot_id,
            run_mode=0,
        )

    expect_failure(
        "prefill_unsupported_width_1",
        lambda: call(wv=torch.randn((1, dim), device=device, dtype=dtype)),
        ("Only support width in",),
    )
    expect_failure(
        "prefill_unsupported_width_5",
        lambda: call(wv=torch.randn((5, dim), device=device, dtype=dtype)),
        ("Only support width in",),
    )
    expect_failure(
        "prefill_dtype_mismatch_weight",
        lambda: call(wv=torch.randn((4, dim), device=device, dtype=torch.float32)),
        ("weight dtype must match",),
    )
    expect_failure(
        "prefill_dtype_mismatch_conv_states",
        lambda: call(csv=torch.randn((4, 3, dim), device=device, dtype=torch.float32)),
        ("conv_states dtype must match",),
    )
    # 2D packed input requires query_start_loc.
    x_2d = torch.randn((2, dim), device=device, dtype=dtype)
    expect_failure(
        "prefill_2d_missing_query_start_loc",
        lambda: call(xv=x_2d),
        ("query_start_loc must have at least 2 elements",),
    )
    expect_failure(
        "prefill_x_rank_not_2_or_3",
        lambda: call(xv=torch.randn((4,), device=device, dtype=dtype)),
        ("x must be 2D or 3D",),
    )
    expect_failure(
        "prefill_weight_rank_not_2",
        lambda: call(wv=torch.randn((4,), device=device, dtype=dtype)),
        ("weight must be 2D",),
    )
    expect_failure(
        "prefill_weight_dim_mismatch",
        lambda: call(wv=torch.randn((4, dim - 1), device=device, dtype=dtype)),
        ("weight last dimension must match x dimension",),
    )
    expect_failure(
        "prefill_conv_states_rank_not_3",
        lambda: call(csv=torch.randn((4, dim), device=device, dtype=dtype)),
        ("conv_states must be 3D",),
    )
    expect_failure(
        "prefill_conv_states_dim_mismatch",
        lambda: call(csv=torch.randn((4, 3, dim - 1), device=device, dtype=dtype)),
        ("conv_states last dimension must match x dimension",),
    )
    expect_failure(
        "prefill_conv_states_too_short",
        lambda: call(csv=torch.randn((4, 2, dim), device=device, dtype=dtype)),
        ("conv_states state length must be at least width - 1",),
    )
    expect_failure(
        "prefill_bias_dim_mismatch",
        lambda: call(bv=torch.randn((dim - 1,), device=device, dtype=dtype)),
        ("bias must be 1D with length equal to x dimension",),
    )
    # Non-contiguous inputs (transpose views).
    expect_failure(
        "prefill_x_not_contiguous",
        lambda: call(xv=torch.randn((2, dim, 4), device=device, dtype=dtype).transpose(1, 2)),
        ("x must be contiguous",),
    )
    expect_failure(
        "prefill_weight_not_contiguous",
        lambda: call(wv=torch.randn((dim, 4), device=device, dtype=dtype).t()),
        ("weight must be contiguous",),
    )
    expect_failure(
        "prefill_conv_states_not_contiguous",
        lambda: call(csv=torch.randn((4, dim, 3), device=device, dtype=dtype).transpose(1, 2)),
        ("conv_states must be contiguous",),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--pad-slot-id", type=int, default=PAD_SLOT_ID)
    args = parser.parse_args()

    try:
        import sgl_kernel_npu  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(f"Import failed: {exc}") from exc

    if not hasattr(torch.ops.npu, "causal_conv1d"):
        raise SystemExit("torch.ops.npu.causal_conv1d is not registered")

    if not hasattr(torch, "npu") or torch.npu.device_count() <= 0:
        raise SystemExit("NPU device is not available")

    torch.manual_seed(args.seed)
    device = torch.device("npu")
    pad_slot_id = args.pad_slot_id

    cases = [
        # 1. Dense baseline (rolling fast path + activation: no bias + SiLU).
        #    Zero history, all has_initial_state=False. width 2, BF16.
        PrefillCaseConfig(
            name="prefill_dense_w2_bf16_nobias_silu_rolling",
            dtype=torch.bfloat16,
            dim=2048,
            width=2,
            state_len=1,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=False,
            input_mode="3d",
            batch=4,
            seq_len=6,
            cache_indices=[2, 0, 3, 1],
            has_initial_state=[False, False, False, False],
        ),
        # 2. Dense mixed state (generic path: bias+SiLU). width 4, FP16,
        #    mixed has_initial_state, state_len > width-1.
        PrefillCaseConfig(
            name="prefill_dense_w4_fp16_bias_silu_mixed_state",
            dtype=torch.float16,
            dim=4096,
            width=4,
            state_len=5,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=3,
            seq_len=4,
            cache_indices=[5, 1, 6],
            has_initial_state=[True, False, True],
        ),
        # 3. Dense channel split + partial last tile: dim=6144 (non-multiple of
        #    4096 -> baseDimCnt=2, last tile 2048).
        PrefillCaseConfig(
            name="prefill_dense_w4_bf16_bias_silu_dim6144_partial_tile",
            dtype=torch.bfloat16,
            dim=6144,
            width=4,
            state_len=5,
            num_cache_lines=6,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=5,
            cache_indices=[0, 3],
            has_initial_state=[True, False],
        ),
        # 4. Packed varlen (rolling+activation in varlen mode): non-uniform
        #    lengths with one L < width-1 (=3) and one longer sequence.
        PrefillCaseConfig(
            name="prefill_packed_w4_bf16_nobias_silu_varlen",
            dtype=torch.bfloat16,
            dim=2048,
            width=4,
            state_len=4,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=False,
            input_mode="2d",
            batch=4,
            lengths=[2, 5, 1, 4],
            cache_indices=[2, 0, 5, 3],
            has_initial_state=[True, False, True, False],
        ),
        # 5. Multi token-block: cuSeqlen=256 (> numCores) -> tokenBlockCnt>=2,
        #    long seq spanning multiple blocks.
        PrefillCaseConfig(
            name="prefill_dense_w4_bf16_bias_silu_multi_token_block",
            dtype=torch.bfloat16,
            dim=1024,
            width=4,
            state_len=5,
            num_cache_lines=20,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=16,
            seq_len=16,
            cache_indices=[0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18],
            has_initial_state=[(i % 2 == 0) for i in range(16)],
        ),
        # 6. Pad slot: one cache_indices == pad_slot_id.
        PrefillCaseConfig(
            name="prefill_dense_w4_bf16_bias_silu_pad_slot",
            dtype=torch.bfloat16,
            dim=2048,
            width=4,
            state_len=3,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=4,
            seq_len=4,
            cache_indices=[0, pad_slot_id, 2, 4],
            has_initial_state=[True, False, True, False],
        ),
    ]

    for case in cases:
        run_prefill_positive_case(
            case,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            pad_slot_id=pad_slot_id,
        )

    run_prefill_negative_cases(
        device=device, dtype=torch.bfloat16, pad_slot_id=pad_slot_id
    )
    print("All causal_conv1d prefill tests passed.")


if __name__ == "__main__":
    main()
