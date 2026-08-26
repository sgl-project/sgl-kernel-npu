import argparse
from dataclasses import dataclass, replace
from typing import Iterable, Optional

import sgl_kernel_npu  # noqa: F401  registers npu ops before pytestmark
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.ops.npu namespace available
from utils import require_npu_op

pytestmark = require_npu_op("causal_conv1d")

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
    query_start_loc_dtype: torch.dtype = torch.int32


def make_query_start_loc(
    lengths: Iterable[int],
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + int(length))
    if device.type == "cpu":
        return torch.tensor(qsl, device="cpu", dtype=dtype)
    out = torch.empty((len(qsl),), device=device, dtype=dtype)
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
    query_start_loc = make_query_start_loc(
        lengths, device, dtype=case.query_start_loc_dtype
    )
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
    query_start_loc = make_query_start_loc(
        lengths, device, dtype=case.query_start_loc_dtype
    )
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
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation_mode=int(case.activation_mode),
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
    base_case = CaseConfig(
        name="negative_case_inputs",
        dtype=dtype,
        dim=4096,
        width=4,
        state_len=5,
        num_cache_lines=8,
        activation_mode=False,
        use_bias=True,
        input_mode="3d",
        batch=2,
        seq_len=4,
        cache_indices=[0, 3],
        has_initial_state=[True, False],
    )
    (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    ) = make_case_tensors(base_case, device, pad_slot_id)

    unsupported_width_case = replace(
        base_case, name="unsupported_width", width=5
    )
    (
        unsupported_x,
        unsupported_weight,
        unsupported_bias,
        unsupported_conv_states,
        unsupported_query_start_loc,
        unsupported_cache_indices,
        unsupported_has_initial_state,
    ) = make_case_tensors(unsupported_width_case, device, pad_slot_id)

    # The native prefill kernel defines MAX_WIDTH=4, so 5 is the first invalid width.
    expect_failure(
        "unsupported_width",
        lambda: torch.ops.npu.causal_conv1d(
            unsupported_x,
            unsupported_weight,
            unsupported_conv_states,
            bias=unsupported_bias,
            query_start_loc=unsupported_query_start_loc,
            cache_indices=unsupported_cache_indices,
            has_initial_state=unsupported_has_initial_state,
        ),
        ("width in [2,4]",),
    )

    expect_failure(
        "missing_query_start_loc_for_varlen",
        lambda: torch.ops.npu.causal_conv1d(
            x.reshape(-1, base_case.dim),
            weight,
            conv_states,
            bias=bias,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        ),
        ("query_start_loc must have at least 2 elements",),
    )

    expect_failure(
        "shape_mismatch_conv_states",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            weight,
            torch.randn((8, 5, 2048), device=device, dtype=dtype),
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        ),
        ("conv_states.shape[2]", "must equal dim"),
    )

    expect_failure(
        "dtype_mismatch_weight",
        lambda: torch.ops.npu.causal_conv1d(
            x,
            torch.randn(
                (base_case.width, base_case.dim),
                device="cpu",
                dtype=torch.float32,
            ).to(device=device),
            conv_states,
            bias=bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        ),
        ("dtype must match",),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--pad-slot-id", type=int, default=PAD_SLOT_ID)
    args = parser.parse_args()

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
            name="dense3d_dim3072",
            dtype=torch.bfloat16,
            dim=3072,
            width=4,
            state_len=5,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=4,
            cache_indices=[0, 3],
            has_initial_state=[True, False],
        ),
        CaseConfig(
            name="dense3d_int64_query_start_loc",
            dtype=torch.bfloat16,
            dim=1024,
            width=4,
            state_len=5,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=4,
            cache_indices=[1, 5],
            has_initial_state=[True, False],
            query_start_loc_dtype=torch.int64,
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
