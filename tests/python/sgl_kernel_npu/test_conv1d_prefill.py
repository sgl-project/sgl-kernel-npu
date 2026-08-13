import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


@dataclass
class CaseConfig:
    name: str
    dtype: torch.dtype
    dim: int
    width: int
    state_len: int
    num_cache_lines: int
    activation_mode: bool
    use_bias: bool
    input_mode: str
    batch: int
    seq_len: Optional[int] = None
    lengths: Optional[list[int]] = None
    cache_indices: Optional[list[int]] = None
    has_initial_state: Optional[list[bool]] = None


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


def make_device_bool_tensor(
    values: Iterable[bool], device: torch.device
) -> torch.Tensor:
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


def make_device_long_tensor(
    values: Iterable[int], device: torch.device
) -> torch.Tensor:
    values = list(values)
    if device.type == "cpu":
        return torch.tensor(values, device="cpu", dtype=torch.int64)
    out = torch.empty((len(values),), device=device, dtype=torch.int64)
    for idx, value in enumerate(values):
        out[idx] = int(value)
    return out


def make_host_bool_tensor(values: Iterable[bool]) -> torch.Tensor:
    return torch.tensor(list(values), device="cpu", dtype=torch.bool)


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x


def reference_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation_mode: bool = False,
    pad_slot_id: int = PAD_SLOT_ID,
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

        cache_idx = int(cache_indices[seq].item())
        if cache_idx == pad_slot_id:
            continue

        valid_mask[start : start + length] = True

        if bool(has_initial_state[seq].item()):
            hist_raw = conv_states[cache_idx, :state_prefix].clone()
        else:
            hist_raw = torch.zeros((state_prefix, dim), device=x.device, dtype=x.dtype)

        x_seg_raw = x_tokens[start : start + length]
        x_ext_raw = torch.cat([hist_raw, x_seg_raw], dim=0)
        x_ext = x_ext_raw.float()

        # generic K-tap causal conv: out[t] = sum_j x_ext[t + j] * weight[j]
        acc = sum(x_ext[j : j + length] * weight_fp32[j] for j in range(width))
        if bias_fp32 is not None:
            acc = acc + bias_fp32
        if activation_mode:
            acc = F.silu(acc)

        y_ref[start : start + length] = acc.to(x.dtype).float()
        conv_states_ref[cache_idx, :state_prefix] = x_ext_raw[-state_prefix:]

    return y_ref, conv_states_ref, valid_mask


def make_case_tensors(case: CaseConfig, device: torch.device, pad_slot_id: int):
    if case.input_mode == "3d":
        assert case.seq_len is not None
        x = torch.randn(
            (case.batch, case.seq_len, case.dim), device=device, dtype=case.dtype
        )
        lengths = [case.seq_len] * case.batch
    else:
        assert case.lengths is not None
        x = torch.randn((sum(case.lengths), case.dim), device=device, dtype=case.dtype)
        lengths = case.lengths

    weight = torch.randn((case.width, case.dim), device=device, dtype=case.dtype)
    bias = (
        torch.randn((case.dim,), device=device, dtype=case.dtype)
        if case.use_bias
        else None
    )
    conv_states = torch.randn(
        (case.num_cache_lines, case.state_len, case.dim),
        device=device,
        dtype=case.dtype,
    )
    query_start_loc = make_query_start_loc(lengths, device)
    cache_indices = make_device_int_tensor(case.cache_indices, device)
    if device.type == "cpu":
        has_initial_state = make_host_bool_tensor(case.has_initial_state)
    else:
        has_initial_state = make_device_bool_tensor(case.has_initial_state, device)

    assert cache_indices.numel() == case.batch
    assert has_initial_state.numel() == case.batch
    assert query_start_loc.numel() == case.batch + 1
    assert (cache_indices == pad_slot_id).sum().item() < case.batch

    return (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


def summarize_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return diff.max().item(), diff.mean().item()


def run_positive_case(
    case: CaseConfig, device: torch.device, atol: float, rtol: float, pad_slot_id: int
):
    host_device = torch.device("cpu")
    (
        x_cpu,
        weight_cpu,
        bias_cpu,
        conv_states_cpu,
        query_start_loc_cpu,
        cache_indices_cpu,
        has_initial_state_cpu,
    ) = make_case_tensors(case, host_device, pad_slot_id)
    lengths = [case.seq_len] * case.batch if case.input_mode == "3d" else case.lengths
    x = x_cpu.to(device=device)
    weight = weight_cpu.to(device=device)
    bias = bias_cpu.to(device=device) if bias_cpu is not None else None
    conv_states_npu = conv_states_cpu.to(device=device)
    query_start_loc = make_query_start_loc(lengths, device)
    cache_indices = make_device_int_tensor(case.cache_indices, device)
    has_initial_state = make_device_bool_tensor(case.has_initial_state, device)

    y_ref, conv_states_ref, valid_mask = reference_causal_conv1d(
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

    y_npu = torch.ops.npu.causal_conv1d(
        x,
        weight,
        conv_states_npu,
        query_start_loc,
        cache_indices,
        has_initial_state,
        bias=bias,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
    )
    torch.npu.synchronize()

    valid_mask_cpu = valid_mask
    y_ref_cpu = y_ref
    y_npu_cpu = flatten_tokens(y_npu).cpu().float()
    conv_states_ref_cpu = conv_states_ref.float()
    conv_states_npu_cpu = conv_states_npu.cpu().float()

    y_ref_valid = y_ref_cpu[valid_mask_cpu]
    y_npu_valid = y_npu_cpu[valid_mask_cpu]
    if y_ref_valid.numel() > 0:
        torch.testing.assert_close(y_npu_valid, y_ref_valid, atol=atol, rtol=rtol)

    torch.testing.assert_close(
        conv_states_npu_cpu, conv_states_ref_cpu, atol=0.0, rtol=0.0
    )

    out_max_abs_diff, out_mean_abs_diff = (
        summarize_diff(y_npu_valid, y_ref_valid)
        if y_ref_valid.numel() > 0
        else (0.0, 0.0)
    )
    state_max_abs_diff, state_mean_abs_diff = summarize_diff(
        conv_states_npu_cpu, conv_states_ref_cpu
    )

    print(
        f"[PASS] {case.name}: "
        f"output(max={out_max_abs_diff:.6g}, mean={out_mean_abs_diff:.6g}) "
        f"state(max={state_max_abs_diff:.6g}, mean={state_mean_abs_diff:.6g})"
    )


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


def run_negative_cases(device: torch.device, dtype: torch.dtype, pad_slot_id: int):
    dim = 4096
    x = torch.randn((2, 4, dim), device=device, dtype=dtype)
    weight = torch.randn((4, dim), device=device, dtype=dtype)
    conv_states = torch.randn((8, 5, dim), device=device, dtype=dtype)
    query_start_loc = make_device_int_tensor([0, 4, 8], device)
    cache_indices = make_device_int_tensor([0, 3], device)
    has_initial_state = make_device_bool_tensor([True, False], device)
    bias = torch.randn((dim,), device=device, dtype=dtype)

    # width 65 would need RS=128 (does not fit UB); widths 2..64 are all supported
    # (any K, routed to the RS = roundUpToPow2(K) variant), so 65 is the first reject.
    expect_failure(
        "unsupported_width",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            torch.randn((65, dim), device=device, dtype=dtype),
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            bias=bias,
        ),
        ("2..64", "width"),
    )

    # NOTE: the previous AscendC kernel only supported a fixed set of dims
    # (e.g. 1024/4096/8192) and rejected others such as 3072. The PTO kernel is
    # generic over `dim` (it tiles the channel axis), so dim=3072 is now a valid
    # case rather than an error; correctness across arbitrary dims is covered by
    # the positive cases and the fp32-reference sweep.
    dim3072 = torch.ops.npu.causal_conv1d(
        torch.randn((2, 4, 3072), device=device, dtype=dtype),
        torch.randn((4, 3072), device=device, dtype=dtype),
        torch.randn((8, 5, 3072), device=device, dtype=dtype),
        query_start_loc,
        cache_indices,
        has_initial_state,
        bias=torch.randn((3072,), device=device, dtype=dtype),
    )
    assert dim3072.shape[-1] == 3072, "PTO causal_conv1d should support dim=3072"
    print("[PASS] supports_dim_3072 (PTO kernel is generic over dim)")

    expect_failure(
        "missing_required_query_start_loc",
        lambda: torch.ops.npu.causal_conv1d(
            x, weight, conv_states, cache_indices, has_initial_state
        ),
        ("missing", "expected at most", "arguments"),
    )

    expect_failure(
        "shape_mismatch_conv_states",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            weight,
            torch.randn((8, 5, 2048), device=device, dtype=dtype),
            query_start_loc,
            cache_indices,
            has_initial_state,
            bias=bias,
        ),
        ("conv_states.shape[2]", "must equal dim"),
    )

    expect_failure(
        "dtype_mismatch_weight",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            torch.randn((4, dim), device="cpu", dtype=torch.float32).to(device=device),
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            bias=bias,
        ),
        ("dtype must match",),
    )

    # [7] wrapper casts int64/bool -> int32/bool, so PyTorch-default dtypes are accepted
    # (int64 query_start_loc used to be rejected).
    y_default_dtypes = torch.ops.npu.causal_conv1d(
        x,
        weight,
        conv_states,
        make_device_long_tensor([0, 4, 8], device),  # int64 query_start_loc
        make_device_long_tensor([0, 3], device),  # int64 cache_indices
        make_device_long_tensor([1, 0], device),  # int64 "has_initial_state" -> bool
        bias=bias,
    )
    torch.npu.synchronize()
    assert (
        y_default_dtypes.shape == x.shape
    ), "int64/bool inputs should be accepted (wrapper casts)"
    print("[PASS] accepts_default_int64_bool_inputs (wrapper casts to int32/bool)")

    # [5] varlen: an empty/too-short query_start_loc would underflow batch -> must reject.
    x2d = torch.randn((4, dim), device=device, dtype=dtype)
    expect_failure(
        "empty_query_start_loc_varlen",
        lambda: torch.ops.npu.causal_conv1d(
            x2d,
            weight,
            conv_states,
            make_device_int_tensor([0], device),  # size 1 (< 2)
            cache_indices,
            has_initial_state,
            bias=bias,
        ),
        ("query_start_loc", "at least 2 elements"),
    )

    # [6] cache_indices / has_initial_state are indexed per sequence -> size must be >= batch (=2).
    expect_failure(
        "cache_indices_too_small",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc,
            make_device_int_tensor([0], device),  # size 1 (< batch 2)
            has_initial_state,
            bias=bias,
        ),
        ("cache_indices", "size >= batch"),
    )
    expect_failure(
        "has_initial_state_too_small",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc,
            cache_indices,
            make_device_bool_tensor([True], device),  # size 1 (< batch 2)
            bias=bias,
        ),
        ("has_initial_state", "size >= batch"),
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

    positive_cases = [
        CaseConfig(
            name="dense3d_all_zero_no_bias",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=3,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=False,
            input_mode="3d",
            batch=2,
            seq_len=6,
            cache_indices=[0, 1],
            has_initial_state=[False, False],
        ),
        CaseConfig(
            name="dense3d_mixed_bias_act",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=5,
            num_cache_lines=24,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=3,
            seq_len=4,
            cache_indices=[5, 12, 20],
            has_initial_state=[True, False, True],
        ),
        CaseConfig(
            name="varlen2d_all_one_bias_act",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=6,
            num_cache_lines=12,
            activation_mode=True,
            use_bias=True,
            input_mode="2d",
            batch=3,
            lengths=[3, 5, 2],
            cache_indices=[2, 4, 8],
            has_initial_state=[True, True, True],
        ),
        CaseConfig(
            name="varlen2d_mixed_pad_no_bias",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=5,
            num_cache_lines=20,
            activation_mode=False,
            use_bias=False,
            input_mode="2d",
            batch=4,
            lengths=[2, 4, 1, 3],
            cache_indices=[3, args.pad_slot_id, 9, 15],
            has_initial_state=[True, False, False, True],
        ),
        CaseConfig(
            name="dense3d_fp16_bias_act",
            dtype=torch.float16,
            dim=1024,
            width=4,
            state_len=4,
            num_cache_lines=10,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=5,
            cache_indices=[1, 7],
            has_initial_state=[True, False],
        ),
        CaseConfig(
            name="dense3d_width3_bias_act",
            dtype=torch.bfloat16,
            dim=2048,
            width=3,
            state_len=4,
            num_cache_lines=12,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=3,
            seq_len=7,
            cache_indices=[0, 5, 9],
            has_initial_state=[True, False, True],
        ),
        CaseConfig(
            name="varlen2d_width2_fp16_act",
            dtype=torch.float16,
            dim=2048,
            width=2,
            state_len=3,
            num_cache_lines=16,
            activation_mode=True,
            use_bias=False,
            input_mode="2d",
            batch=4,
            lengths=[3, 5, 1, 4],
            cache_indices=[1, 6, args.pad_slot_id, 11],
            has_initial_state=[True, True, False, True],
        ),
        CaseConfig(
            name="dense3d_width5_bias_act",
            dtype=torch.bfloat16,
            dim=2048,
            width=5,
            state_len=5,
            num_cache_lines=12,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=9,
            cache_indices=[0, 6],
            has_initial_state=[True, True],
        ),
        CaseConfig(
            name="varlen2d_width8_fp16_act",
            dtype=torch.float16,
            dim=6144,
            width=8,
            state_len=8,
            num_cache_lines=16,
            activation_mode=True,
            use_bias=True,
            input_mode="2d",
            batch=3,
            lengths=[10, 4, 7],
            cache_indices=[2, 9, 14],
            has_initial_state=[True, False, True],
        ),
        CaseConfig(
            name="dense3d_width16_bias_act",
            dtype=torch.bfloat16,
            dim=2048,
            width=16,
            state_len=16,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=20,
            cache_indices=[0, 4],
            has_initial_state=[True, False],
        ),
        CaseConfig(
            name="dense3d_width32_fp16_act",
            dtype=torch.float16,
            dim=2048,
            width=32,
            state_len=32,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=False,
            input_mode="3d",
            batch=2,
            seq_len=40,
            cache_indices=[1, 5],
            has_initial_state=[True, True],
        ),
        CaseConfig(
            name="varlen2d_width64_bias_act",
            dtype=torch.bfloat16,
            dim=2048,
            width=64,
            state_len=64,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            input_mode="2d",
            batch=2,
            lengths=[70, 50],
            cache_indices=[0, 3],
            has_initial_state=[True, False],
        ),
        # Non-power-of-two widths: K is passed at runtime and routed to the
        # RS = roundUpToPow2(K) variant, so these exercise K < RS in each RS group
        # (7->rs8, 12->rs16, 24->rs32, 48->rs64). widths 3 and 5 above cover rs4/rs8.
        CaseConfig(
            name="dense3d_width7_fp16_bias_act",
            dtype=torch.float16,
            dim=2048,
            width=7,
            state_len=7,
            num_cache_lines=12,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=12,
            cache_indices=[0, 5],
            has_initial_state=[True, False],
        ),
        CaseConfig(
            name="varlen2d_width12_bias_act",
            dtype=torch.bfloat16,
            dim=2048,
            width=12,
            state_len=12,
            num_cache_lines=10,
            activation_mode=True,
            use_bias=True,
            input_mode="2d",
            batch=2,
            lengths=[20, 14],
            cache_indices=[0, 4],
            has_initial_state=[True, True],
        ),
        CaseConfig(
            name="dense3d_width24_fp16_act",
            dtype=torch.float16,
            dim=2048,
            width=24,
            state_len=24,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=False,
            input_mode="3d",
            batch=2,
            seq_len=30,
            cache_indices=[1, 5],
            has_initial_state=[True, True],
        ),
        CaseConfig(
            name="varlen2d_width48_bias_act",
            dtype=torch.bfloat16,
            dim=2048,
            width=48,
            state_len=48,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            input_mode="2d",
            batch=2,
            lengths=[60, 50],
            cache_indices=[0, 3],
            has_initial_state=[True, False],
        ),
    ]

    for case in positive_cases:
        run_positive_case(
            case,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            pad_slot_id=args.pad_slot_id,
        )

    run_negative_cases(
        device=device, dtype=torch.bfloat16, pad_slot_id=args.pad_slot_id
    )
    print("All causal_conv1d prefill tests passed.")


if __name__ == "__main__":
    main()
