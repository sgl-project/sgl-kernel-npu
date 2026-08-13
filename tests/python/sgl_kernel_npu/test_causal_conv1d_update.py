"""Tests for fused causal_conv1d update/decode and MTP (run_mode=1)."""

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


# CPU reference: update always reads the history prefix; has_initial_state is ignored.


def reference_causal_conv1d_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
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

        # Valid cache indices are unique within one invocation. Keep the reference
        # state evolving across sequences to model successive host-side calls.
        hist = conv_states_ref[cache_idx, :state_prefix].clone()
        x_seg = x_tokens[start : start + length]
        ext_raw = torch.cat([hist, x_seg], dim=0)
        ext = ext_raw.float()

        acc = sum(ext[j : j + length] * weight_fp32[j] for j in range(width))
        if bias_fp32 is not None:
            acc = acc + bias_fp32
        if activation_mode:
            acc = F.silu(acc)

        y_ref[start : start + length] = acc.to(x.dtype).float()
        # State writeback is a plain copy of the last (width-1) tokens; keep the
        # original dtype so the comparison against the kernel's DataCopy is exact.
        conv_states_ref[cache_idx, :state_prefix] = ext_raw[-state_prefix:]

    return y_ref, conv_states_ref, valid_mask


@dataclass
class UpdateCaseConfig:
    name: str
    dtype: torch.dtype
    dim: int
    width: int
    state_len: int
    num_cache_lines: int
    activation_mode: bool
    use_bias: bool
    batch: int
    cache_indices: list[int]
    steps: int = 3


def run_update_positive_case(
    case: UpdateCaseConfig, device: torch.device, atol: float, rtol: float, pad_slot_id: int
):
    host_device = torch.device("cpu")

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
    assert len(case.cache_indices) == case.batch
    assert (cache_indices_cpu == pad_slot_id).sum().item() < case.batch

    # Pre-generate the decode token stream on CPU so the reference and the NPU
    # op see identical inputs at every step.
    x_steps_cpu = torch.randn(
        (case.batch, case.steps, case.dim), device=host_device, dtype=case.dtype
    )

    weight = weight_cpu.to(device=device)
    bias = bias_cpu.to(device=device) if bias_cpu is not None else None
    cache_indices = make_device_int_tensor(case.cache_indices, device)

    conv_states_ref = conv_states_cpu.clone()
    conv_states_npu = conv_states_cpu.to(device=device)

    # Snapshot unreferenced/padded rows to assert they are never touched.
    referenced_rows = {
        int(ci) for ci in case.cache_indices if ci != pad_slot_id and 0 <= ci < case.num_cache_lines
    }
    untouched_rows = sorted(set(range(case.num_cache_lines)) - referenced_rows)
    untouched_snapshot = (
        conv_states_cpu[untouched_rows].clone() if untouched_rows else None
    )

    for step in range(case.steps):
        x_step_cpu = x_steps_cpu[:, step, :].unsqueeze(1)  # [B, 1, D]

        y_ref, conv_states_ref, valid_mask = reference_causal_conv1d_update(
            x=x_step_cpu,
            weight=weight_cpu,
            conv_states=conv_states_ref,
            query_start_loc=None,
            cache_indices=cache_indices_cpu,
            bias=bias_cpu,
            activation_mode=case.activation_mode,
            pad_slot_id=pad_slot_id,
        )

        x_step = x_step_cpu.to(device=device)
        y_npu = torch.ops.npu.causal_conv1d(
            x=x_step,
            weight=weight,
            conv_states=conv_states_npu,
            bias=bias,
            cache_indices=cache_indices,
            activation_mode=case.activation_mode,
            pad_slot_id=pad_slot_id,
            run_mode=1,
        )
        torch.npu.synchronize()

        y_npu_cpu = flatten_tokens(y_npu).cpu().float()
        y_ref_cpu = y_ref
        y_ref_valid = y_ref_cpu[valid_mask]
        y_npu_valid = y_npu_cpu[valid_mask]
        if y_ref_valid.numel() > 0:
            torch.testing.assert_close(y_npu_valid, y_ref_valid, atol=atol, rtol=rtol)

        # State writeback is a plain DataCopy -> require bit-exact match.
        torch.testing.assert_close(
            conv_states_npu.cpu().float(),
            conv_states_ref.float(),
            atol=0.0,
            rtol=0.0,
        )

    if untouched_snapshot is not None:
        torch.testing.assert_close(
            conv_states_npu[untouched_rows].cpu().float(),
            untouched_snapshot.float(),
            atol=0.0,
            rtol=0.0,
        )

    print(f"[PASS] {case.name}: {case.steps}-step decode (width={case.width})")


def run_update_negative_cases(device: torch.device, dtype: torch.dtype, pad_slot_id: int):
    dim = 4096
    x = torch.randn((2, 1, dim), device=device, dtype=dtype)
    weight = torch.randn((4, dim), device=device, dtype=dtype)
    conv_states = torch.randn((4, 3, dim), device=device, dtype=dtype)
    cache_indices = make_device_int_tensor([0, 2], device)
    bias = torch.randn((dim,), device=device, dtype=dtype)

    expect_failure(
        "update_unsupported_width",
        lambda: torch.ops.npu.causal_conv1d(
            x=x,
            weight=torch.randn((5, dim), device=device, dtype=dtype),
            conv_states=conv_states,
            bias=bias,
            cache_indices=cache_indices,
            run_mode=1,
        ),
        ("Only support width in",),
    )

    expect_failure(
        "update_dtype_mismatch_weight",
        lambda: torch.ops.npu.causal_conv1d(
            x=x,
            weight=torch.randn((4, dim), device=device, dtype=torch.float32),
            conv_states=conv_states,
            bias=bias,
            cache_indices=cache_indices,
            run_mode=1,
        ),
        ("weight dtype must match",),
    )

    expect_failure(
        "update_dtype_mismatch_conv_states",
        lambda: torch.ops.npu.causal_conv1d(
            x=x,
            weight=weight,
            conv_states=torch.randn((4, 3, dim), device=device, dtype=torch.float32),
            bias=bias,
            cache_indices=cache_indices,
            run_mode=1,
        ),
        ("conv_states dtype must match",),
    )

    # 2D packed input in update mode requires query_start_loc.
    x_2d = torch.randn((2, dim), device=device, dtype=dtype)
    expect_failure(
        "update_2d_missing_query_start_loc",
        lambda: torch.ops.npu.causal_conv1d(
            x=x_2d,
            weight=weight,
            conv_states=conv_states,
            bias=bias,
            cache_indices=cache_indices,
            run_mode=1,
        ),
        ("query_start_loc must have at least 2 elements",),
    )


# CPU reference for the width-4 MTP branch.


def reference_causal_conv1d_mtp_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation_mode: bool,
    pad_slot_id: int,
):
    width = weight.shape[0]
    assert width == 4, "MTP spec path is only entered for width==4"
    state_prefix = width - 1
    keep = width - 2
    dim = x.shape[-1]
    x_tokens = flatten_tokens(x)
    batch = query_start_loc.numel() - 1
    state_len = conv_states.shape[1]
    max_offset = state_len - state_prefix

    y_ref = torch.zeros((x_tokens.shape[0], dim), device=x.device, dtype=torch.float32)
    valid_mask = torch.zeros((x_tokens.shape[0],), device="cpu", dtype=torch.bool)
    conv_states_ref = conv_states.clone()

    weight_fp32 = weight.float()
    bias_fp32 = bias.float() if bias is not None else None

    for seq in range(batch):
        start = int(query_start_loc[seq].item())
        end = int(query_start_loc[seq + 1].item())
        length = end - start
        if length <= 0:
            continue

        cache_idx = int(cache_indices[seq].item())
        if cache_idx == pad_slot_id:
            continue

        accepted = int(num_accepted_tokens[seq].item())
        offset = accepted - 1
        if offset < 0:
            offset = 0
        elif offset > max_offset:
            offset = max_offset

        valid_mask[start : start + length] = True

        hist = conv_states_ref[cache_idx, offset : offset + state_prefix].clone()
        x_seg = x_tokens[start : start + length]
        ext_raw = torch.cat([hist, x_seg], dim=0)
        ext = ext_raw.float()

        acc = sum(ext[j : j + length] * weight_fp32[j] for j in range(width))
        if bias_fp32 is not None:
            acc = acc + bias_fp32
        if activation_mode:
            acc = F.silu(acc)
        y_ref[start : start + length] = acc.to(x.dtype).float()

        if keep + length <= state_len:
            # Spec writeback: shift the two tokens past the history window to the
            # front, then append the freshly committed tokens. Clone the sources
            # so the assignments cannot alias each other within the same row.
            src0 = conv_states_ref[cache_idx, offset + 1].clone()
            src1 = conv_states_ref[cache_idx, offset + 2].clone()
            conv_states_ref[cache_idx, 0] = src0
            conv_states_ref[cache_idx, 1] = src1
            conv_states_ref[cache_idx, 2 : 2 + length] = x_seg
        else:
            # Fallback to the plain WriteBackState path.
            conv_states_ref[cache_idx, :state_prefix] = ext_raw[-state_prefix:]

    return y_ref, conv_states_ref, valid_mask


@dataclass
class MtpCaseConfig:
    name: str
    dtype: torch.dtype
    dim: int
    state_len: int
    num_cache_lines: int
    activation_mode: bool
    use_bias: bool
    accepted: list[int]  # one per batch; length==batch; drives per-seq len==accepted
    cache_indices: list[int]


def run_mtp_positive_case(
    case: MtpCaseConfig, device: torch.device, atol: float, rtol: float, pad_slot_id: int
):
    host_device = torch.device("cpu")
    width = 4
    state_prefix = width - 1
    batch = len(case.accepted)

    assert all(a >= 0 for a in case.accepted), "accepted must be non-negative"
    assert case.state_len >= state_prefix, "state_len must be >= width-1"
    assert len(case.cache_indices) == batch
    assert max(case.cache_indices) < case.num_cache_lines

    lengths = case.accepted  # per-seq len == accepted (the well-defined MTP layout)
    total = sum(lengths)

    weight_cpu = torch.randn((width, case.dim), device=host_device, dtype=case.dtype)
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
    x_cpu = torch.randn((total, case.dim), device=host_device, dtype=case.dtype)
    query_start_loc_cpu = make_query_start_loc(lengths, host_device)
    cache_indices_cpu = make_device_int_tensor(case.cache_indices, host_device)
    num_accepted_cpu = make_device_int_tensor(case.accepted, host_device)

    y_ref, conv_states_ref, valid_mask = reference_causal_conv1d_mtp_update(
        x=x_cpu,
        weight=weight_cpu,
        conv_states=conv_states_cpu,
        query_start_loc=query_start_loc_cpu,
        cache_indices=cache_indices_cpu,
        num_accepted_tokens=num_accepted_cpu,
        bias=bias_cpu,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
    )

    weight = weight_cpu.to(device=device)
    bias = bias_cpu.to(device=device) if bias_cpu is not None else None
    conv_states_npu = conv_states_cpu.to(device=device)
    x = x_cpu.to(device=device)
    query_start_loc = make_query_start_loc(lengths, device)
    cache_indices = make_device_int_tensor(case.cache_indices, device)
    num_accepted = make_device_int_tensor(case.accepted, device)

    y_npu = torch.ops.npu.causal_conv1d(
        x=x,
        weight=weight,
        conv_states=conv_states_npu,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
        run_mode=1,
    )
    torch.npu.synchronize()

    y_npu_cpu = flatten_tokens(y_npu).cpu().float()
    y_ref_valid = y_ref[valid_mask]
    y_npu_valid = y_npu_cpu[valid_mask]
    if y_ref_valid.numel() > 0:
        torch.testing.assert_close(y_npu_valid, y_ref_valid, atol=atol, rtol=rtol)

    torch.testing.assert_close(
        conv_states_npu.cpu().float(),
        conv_states_ref.float(),
        atol=0.0,
        rtol=0.0,
    )

    # Unreferenced rows must be byte-for-byte unchanged.
    referenced_rows = {
        int(ci) for ci in case.cache_indices if ci != pad_slot_id and 0 <= ci < case.num_cache_lines
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
        f"[PASS] {case.name}: mtp accepted={case.accepted} "
        f"output(max={out_max:.6g}, mean={out_mean:.6g})"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


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

    update_cases = [
        UpdateCaseConfig(
            name="decode_w4_bf16_bias_silu_state3",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=3,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            batch=4,
            cache_indices=[3, 1, 4, 0],
        ),
        UpdateCaseConfig(
            name="decode_w4_fp16_nobias_nosilu_state5",
            dtype=torch.float16,
            dim=2048,
            width=4,
            state_len=5,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=False,
            batch=3,
            cache_indices=[2, 0, 5],
        ),
        UpdateCaseConfig(
            name="decode_w3_bf16_bias_nosilu_state4_dim8192",
            dtype=torch.bfloat16,
            dim=8192,
            width=3,
            state_len=4,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=True,
            batch=4,
            cache_indices=[5, 2, 7, 1],
        ),
        UpdateCaseConfig(
            name="decode_w2_fp16_nobias_silu_state2",
            dtype=torch.float16,
            dim=1024,
            width=2,
            state_len=2,
            num_cache_lines=6,
            activation_mode=True,
            use_bias=False,
            batch=3,
            cache_indices=[1, 4, 2],
        ),
        UpdateCaseConfig(
            name="decode_w4_bf16_bias_silu_pad_slot",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=3,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=True,
            batch=4,
            cache_indices=[0, args.pad_slot_id, 2, 5],
        ),
        UpdateCaseConfig(
            name="decode_w4_fp16_bias_nosilu_state6_batch6",
            dtype=torch.float16,
            dim=2048,
            width=4,
            state_len=6,
            num_cache_lines=10,
            activation_mode=False,
            use_bias=True,
            batch=6,
            cache_indices=[7, 3, 0, 5, 1, 8],
        ),
    ]

    for case in update_cases:
        run_update_positive_case(
            case,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            pad_slot_id=args.pad_slot_id,
        )

    run_update_negative_cases(
        device=device, dtype=torch.bfloat16, pad_slot_id=args.pad_slot_id
    )

    # MTP uses packed sequences with per-sequence accepted-token counts.
    mtp_cases = [
        MtpCaseConfig(
            name="mtp_w4_bf16_bias_silu_full",
            dtype=torch.bfloat16,
            dim=4096,
            state_len=6,
            num_cache_lines=10,
            activation_mode=True,
            use_bias=True,
            # accepted=0 -> empty seq, verifies skip path only (not spec semantics).
            accepted=[1, 2, 3, 4, 0, 5],
            cache_indices=[2, 0, 5, 7, 1, 3],
        ),
        MtpCaseConfig(
            name="mtp_w4_fp16_nobias_nosilu",
            dtype=torch.float16,
            dim=2048,
            state_len=6,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=False,
            accepted=[2, 4, 1],
            cache_indices=[4, 0, 6],
        ),
        MtpCaseConfig(
            name="mtp_w4_bf16_nobias_silu_boundary",
            dtype=torch.bfloat16,
            dim=2048,
            state_len=4,
            num_cache_lines=8,
            activation_mode=True,
            use_bias=False,
            # L=4 -> max_offset=1; accepted=1,2,3(L-1 fallback); 0(empty-seq skip)
            accepted=[1, 2, 3, 0],
            cache_indices=[3, 0, 5, 1],
        ),
    ]

    for case in mtp_cases:
        run_mtp_positive_case(
            case,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            pad_slot_id=args.pad_slot_id,
        )

    print("All causal_conv1d update/decode tests passed.")


if __name__ == "__main__":
    main()
