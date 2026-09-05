"""A5 FusedDeepMoe correctness and performance comparison script."""

import argparse
import os
import random
import traceback
from typing import Dict, List, Tuple

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from utils import calc_diff, init_dist, profile_npu_event_sequences

torch_npu.npu.config.allow_internal_format = True

MX_QUANT_CONFIGS = {
    "fp8_e4m3": {
        # In torch_npu/op-plugin npu_moe_distribute_dispatch_v2, quant_mode=4
        # selects the generic MX branch; the actual MX output format is then
        # determined by y_dtype (FP8 here, FP4 in the other config below).
        "dispatch_quant_mode": 4,
        "dispatch_y_dtype": torch.float8_e4m3fn,
        "quant_dst_type": torch.float8_e4m3fn,
        "origin_dtype": torch.float8_e4m3fn,
    },
    "fp8_e5m2": {
        "dispatch_quant_mode": 4,
        "dispatch_y_dtype": torch.float8_e5m2,
        "quant_dst_type": torch.float8_e5m2,
        "origin_dtype": torch.float8_e5m2,
    },
    "fp4_e2m1": {
        "dispatch_quant_mode": 4,
        "dispatch_y_dtype": torch_npu.float4_e2m1fn_x2,
        "quant_dst_type": torch_npu.float4_e2m1fn_x2,
        "origin_dtype": torch.float4_e2m1fn_x2,
    },
}
# On A5 fused_deep_moe the actual MX quant path is inferred from the passed
# quantized weight dtype. The Python API still has a quant_mode slot, so keep a
# fixed compatibility value here instead of treating it as a user-facing switch.
FUSED_COMPAT_QUANT_MODE = 0

FUSED_EVENT_PATTERNS = (("FusedDeepMoe", "aclnnFusedDeepMoe"),)
SMALL_OP_EVENT_PATTERNS = (
    ("MoeDistributeDispatchV2", "MoeDistributeDispatchV3"),
    (
        "GroupedMatmul",
        "GroupedMatMul",
        "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul",
    ),
    ("Swiglu", "SwiGlu"),
    ("DynamicMxQuant", "DynamicMXQuant"),
    (
        "GroupedMatmul",
        "GroupedMatMul",
        "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul",
    ),
    ("MoeDistributeCombineV2", "MoeDistributeCombineV3"),
)
SMALL_OP_EVENT_LABELS = ("dispatch", "gmm1", "swiglu", "requant", "gmm2", "combine")
# Only the very first profiler warmup iteration gets this heavier burn-in.
# The extra work is intentionally excluded from final statistics and only
# exists to keep the device busy a bit longer before the profiled small-op
# iterations start.
SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS = 100
# The burn-in matmul uses larger tensors than the routed path on purpose.
# Increasing dimensions tends to create a more stable device-side delay than
# simply increasing the repeat count of very small kernels.
SMALL_FIRST_WARMUP_MATMUL_DIM_SCALE = 4
ACCURACY_ATOL = 2.0
ACCURACY_RTOL = 0.02
PROFILE_DEBUG_HEAD_BYTES = 32


def get_mx_quant_config(args: argparse.Namespace) -> Dict[str, object]:
    return MX_QUANT_CONFIGS[args.quant]


def maybe_cast_weight_to_nz(
    args: argparse.Namespace,
    weight: torch.Tensor,
    origin_dtype: torch.dtype,
    logical_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Convert a quantized weight to A5 FRACTAL_NZ storage.

    FP4 quantization returns packed bytes whose subsequent view as float4 is a
    recovery-format tensor.  CANN does not accept that recovery format as the
    input to npu_format_cast, so the layout conversion must happen first.
    """
    if args.weight_format != "NZ":
        return weight.view(origin_dtype)

    # Prefer the public enum when available; keep numeric 29 as a compatibility
    # fallback for torch_npu versions that do not expose Format.FRACTAL_NZ.
    npu_format = getattr(torch_npu, "Format", None)
    fractal_nz = getattr(npu_format, "FRACTAL_NZ", 29)
    try:
        if args.quant == "fp4_e2m1":
            weight = torch_npu.npu_format_cast(weight, fractal_nz)
        else:
            weight = torch_npu.npu_format_cast(weight.view(origin_dtype), fractal_nz)
    except RuntimeError as exc:
        if args.quant == "fp4_e2m1":
            raise RuntimeError(
                "FP4-NZ input construction is unavailable in this CANN/PTA "
                "environment: packed FP4 npu_format_cast failed. "
                "The fused kernel was not invoked."
            ) from exc
        raise

    weight = weight.view(origin_dtype)
    format_code = torch_npu.get_npu_format(weight)
    if format_code != fractal_nz:
        raise RuntimeError(
            f"FP4/quantized NZ construction produced format {format_code}, "
            f"expected FRACTAL_NZ({fractal_nz})"
        )
    if args.quant == "fp4_e2m1":
        expected_bytes = (
            ((logical_shape[2] + 63) // 64)
            * ((logical_shape[1] + 15) // 16)
            * 16
            * 64
            // 2
            * logical_shape[0]
        )
        actual_bytes = weight.untyped_storage().nbytes()
        if actual_bytes < expected_bytes:
            raise RuntimeError(
                f"FP4-NZ storage is too small: got {actual_bytes} bytes, "
                f"expected at least {expected_bytes} for logical shape {logical_shape}"
            )
    return weight


def log_quant_tensor(rank: int, enabled: bool, name: str, tensor: torch.Tensor):
    if enabled and rank == 0:
        print(
            f"[quant-dtype] {name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}",
            flush=True,
        )


def get_npu_format_desc(tensor: torch.Tensor) -> str:
    format_code = torch_npu.get_npu_format(tensor)
    format_name_map = {
        2: "ND",
        29: "FRACTAL_NZ",
    }
    format_name = format_name_map.get(format_code, f"UNKNOWN_{format_code}")
    return f"{format_name}({format_code})"


def log_tensor_meta(rank: int, enabled: bool, name: str, tensor: torch.Tensor):
    if enabled and rank == 0:
        print(
            f"[tensor-meta] {name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
            f"format={get_npu_format_desc(tensor)}",
            flush=True,
        )


def make_umdk_static_inputs(
    rank: int,
    world_size: int,
    local_num_tokens: int,
    token_prefix_before_rank: int,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    quant_cfg = get_mx_quant_config(args)
    assert args.num_experts % world_size == 0
    local_experts = args.num_experts // world_size

    x = torch.rand((local_num_tokens, args.hidden)).bfloat16().npu() * 2 - 1
    if args.single_active_rank is not None and local_num_tokens > 0:
        topk_rng = random.Random(args.single_active_topk_seed + rank)
        sampled_rows = [
            torch.tensor(
                topk_rng.sample(range(args.num_experts), args.num_topk),
                dtype=torch.int32,
            )
            for _ in range(local_num_tokens)
        ]
        expert_ids = torch.stack(sampled_rows, dim=0).npu()
    else:
        expert_ids = torch.arange(
            token_prefix_before_rank * args.num_topk,
            (token_prefix_before_rank + local_num_tokens) * args.num_topk,
            dtype=torch.int32,
        ).view(local_num_tokens, args.num_topk)
        expert_ids = expert_ids.remainder(args.num_experts).npu()
    expert_scales = torch.rand(
        (local_num_tokens, args.num_topk), dtype=torch.float32
    ).npu()

    gmm1_fp = (
        torch.rand((local_experts, args.hidden, args.moe_intermediate_size * 2))
        .bfloat16()
        .npu()
        * 2
        - 1
    )
    gmm1_weight, gmm1_scale_raw = torch_npu.npu_dynamic_mx_quant(
        gmm1_fp, dst_type=quant_cfg["quant_dst_type"], axis=1
    )
    gmm1_weight = maybe_cast_weight_to_nz(
        args,
        gmm1_weight,
        quant_cfg["origin_dtype"],
        (local_experts, args.hidden, args.moe_intermediate_size * 2),
    )
    gmm1_scale = gmm1_scale_raw.view(torch.float8_e8m0fnu)

    gmm2_fp = (
        torch.rand((local_experts, args.moe_intermediate_size, args.hidden))
        .bfloat16()
        .npu()
        * 2
        - 1
    )
    gmm2_weight, gmm2_scale_raw = torch_npu.npu_dynamic_mx_quant(
        gmm2_fp, dst_type=quant_cfg["quant_dst_type"], axis=1
    )
    gmm2_weight = maybe_cast_weight_to_nz(
        args,
        gmm2_weight,
        quant_cfg["origin_dtype"],
        (local_experts, args.moe_intermediate_size, args.hidden),
    )
    gmm2_scale = gmm2_scale_raw.view(torch.float8_e8m0fnu)

    return {
        "x": x,
        "expert_ids": expert_ids,
        "expert_scales": expert_scales,
        "gmm1_weight_q": gmm1_weight,
        "gmm1_weight_scale": gmm1_scale,
        "gmm2_weight_q": gmm2_weight,
        "gmm2_weight_scale": gmm2_scale,
    }


def make_small_warmup_burn_in_buffers(
    local_num_tokens: int, args: argparse.Namespace
) -> Tuple[torch.Tensor, torch.Tensor]:
    # These tensors are dedicated to the first warmup burn-in path and are not
    # consumed by the routed MoE computation. Keeping them separate makes the
    # functional path easier to reason about.
    dim_scale = SMALL_FIRST_WARMUP_MATMUL_DIM_SCALE
    lhs_rows = local_num_tokens * dim_scale
    shared_dim = args.hidden * dim_scale
    rhs_cols = args.moe_intermediate_size * dim_scale
    lhs = torch.rand((lhs_rows, shared_dim)).bfloat16().npu() * 2 - 1
    rhs = torch.rand((shared_dim, rhs_cols)).bfloat16().npu() * 2 - 1
    return lhs, rhs


def make_small_op_padded_inputs(
    inputs: Dict[str, torch.Tensor],
    local_num_tokens: int,
    padded_num_tokens: int,
) -> Dict[str, torch.Tensor]:
    x_padded = torch.zeros(
        (padded_num_tokens, inputs["x"].shape[1]),
        dtype=inputs["x"].dtype,
        device=inputs["x"].device,
    )
    expert_ids_padded = torch.zeros(
        (padded_num_tokens, inputs["expert_ids"].shape[1]),
        dtype=inputs["expert_ids"].dtype,
        device=inputs["expert_ids"].device,
    )
    expert_scales_padded = torch.zeros(
        (padded_num_tokens, inputs["expert_scales"].shape[1]),
        dtype=inputs["expert_scales"].dtype,
        device=inputs["expert_scales"].device,
    )
    x_active_mask = torch.zeros(
        (padded_num_tokens,), dtype=torch.bool, device=inputs["x"].device
    )

    x_padded[:local_num_tokens] = inputs["x"]
    expert_ids_padded[:local_num_tokens] = inputs["expert_ids"]
    expert_scales_padded[:local_num_tokens] = inputs["expert_scales"]
    x_active_mask[:local_num_tokens] = True

    return {
        **inputs,
        "x": x_padded,
        "expert_ids": expert_ids_padded,
        "expert_scales": expert_scales_padded,
        "x_active_mask": x_active_mask,
        "padded_num_tokens": padded_num_tokens,
    }


def make_random_profile_cases(
    base_inputs: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    local_num_tokens: int,
    padded_num_tokens: int,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    cases = []
    total_case_count = args.num_warmups + args.num_tests
    for case_idx in range(total_case_count):
        case_seed = args.random_profile_seed + case_idx * world_size + rank
        torch_generator = torch.Generator(device="npu")
        torch_generator.manual_seed(case_seed)
        routing_rng = random.Random(case_seed)

        x = (
            torch.rand(
                (local_num_tokens, args.hidden),
                dtype=torch.float32,
                device="npu",
                generator=torch_generator,
            ).bfloat16()
            * 2
            - 1
        )
        expert_scales = torch.rand(
            (local_num_tokens, args.num_topk),
            dtype=torch.float32,
            device="npu",
            generator=torch_generator,
        )
        if local_num_tokens > 0:
            sampled_rows = [
                routing_rng.sample(range(args.num_experts), args.num_topk)
                for _ in range(local_num_tokens)
            ]
            expert_ids = torch.tensor(sampled_rows, dtype=torch.int32).npu()
        else:
            expert_ids = torch.empty((0, args.num_topk), dtype=torch.int32).npu()

        fused_inputs = {
            **base_inputs,
            "x": x,
            "expert_ids": expert_ids,
            "expert_scales": expert_scales,
        }
        cases.append(
            {
                "seed": case_seed,
                "fused_inputs": fused_inputs,
                "small_inputs": make_small_op_padded_inputs(
                    fused_inputs, local_num_tokens, padded_num_tokens
                ),
            }
        )
    return cases


def make_fixed_profile_cases(
    base_inputs: Dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    local_num_tokens: int,
    padded_num_tokens: int,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Create distinct tensor storage with identical values for every profile call.

    This mode separates cross-iteration state corruption from input-dependent
    numerical differences.  We intentionally clone only the per-call inputs;
    quantized weights remain shared and immutable.
    """
    total_case_count = args.num_warmups + args.num_tests
    source = make_random_profile_cases(
        base_inputs,
        rank,
        world_size,
        local_num_tokens,
        padded_num_tokens,
        argparse.Namespace(**{**vars(args), "num_warmups": 1, "num_tests": 0}),
    )[0]
    cases = []
    for case_idx in range(total_case_count):
        fused_inputs = dict(source["fused_inputs"])
        for name in ("x", "expert_ids", "expert_scales"):
            fused_inputs[name] = source["fused_inputs"][name].clone()
        small_inputs = make_small_op_padded_inputs(
            fused_inputs, local_num_tokens, padded_num_tokens
        )
        cases.append(
            {
                "seed": source["seed"],
                "fused_inputs": fused_inputs,
                "small_inputs": small_inputs,
                "fixed_values": True,
                "case_index": case_idx,
            }
        )
    return cases


def assert_unique_tensor_storage(tensors: List[torch.Tensor], name: str):
    data_ptrs = [tensor.data_ptr() for tensor in tensors if tensor.numel() > 0]
    if len(data_ptrs) != len(set(data_ptrs)):
        raise AssertionError(
            f"{name} must use distinct storage for every profile iteration"
        )


def format_tensor_head_bytes(
    tensor: torch.Tensor, max_bytes: int = PROFILE_DEBUG_HEAD_BYTES
) -> str:
    """Return a compact raw-byte preview after all profiled work is complete."""
    try:
        byte_tensor = tensor.detach().contiguous().cpu().view(torch.uint8).flatten()
        values = byte_tensor[:max_bytes].tolist()
        return " ".join(f"{int(value):02x}" for value in values)
    except Exception as exc:
        return f"<unavailable: {exc}>"


def print_profile_case_head_bytes(
    profile_cases: List[Dict[str, object]],
    small_results: List[Tuple[torch.Tensor, torch.Tensor]],
    fused_results: List[Tuple[torch.Tensor, torch.Tensor]],
    rank: int,
):
    """Print compact per-iteration input/output bytes for a failed profile."""
    print(
        f"[rank {rank}] profile debug tensor heads "
        f"(first {PROFILE_DEBUG_HEAD_BYTES} bytes; values are raw storage bytes):",
        flush=True,
    )
    input_names = ("x", "expert_ids", "expert_scales")
    case_count = min(len(profile_cases), len(small_results), len(fused_results))
    for case_idx in range(case_count):
        case = profile_cases[case_idx]
        print(
            f"[rank {rank}] iteration={case_idx}, seed={case.get('seed')}, "
            f"fixed={case.get('fixed_values', False)}",
            flush=True,
        )
        for side, inputs in (
            ("small_input", case.get("small_inputs", {})),
            ("fused_input", case.get("fused_inputs", {})),
        ):
            for input_name in input_names:
                tensor = inputs.get(input_name)
                if tensor is None:
                    continue
                print(
                    f"  {side}.{input_name}: dtype={tensor.dtype}, "
                    f"shape={tuple(tensor.shape)}, data_ptr=0x{tensor.data_ptr():x}, "
                    f"bytes={format_tensor_head_bytes(tensor)}",
                    flush=True,
                )
        small_output, small_counts = small_results[case_idx]
        fused_output, fused_counts = fused_results[case_idx]
        for side, tensor in (
            ("small_output", small_output),
            ("fused_output", fused_output),
            ("small_counts", small_counts),
            ("fused_counts", fused_counts),
        ):
            print(
                f"  {side}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
                f"data_ptr=0x{tensor.data_ptr():x}, "
                f"bytes={format_tensor_head_bytes(tensor)}",
                flush=True,
            )


def validate_random_profile_results(
    profile_cases: List[Dict[str, object]],
    small_results: List[Tuple[torch.Tensor, torch.Tensor]],
    fused_results: List[Tuple[torch.Tensor, torch.Tensor]],
    rank: int,
    world_size: int,
    local_num_tokens: int,
    padded_num_tokens: int,
    local_experts: int,
    args: argparse.Namespace,
):
    local_error = None
    total_case_count = args.num_warmups + args.num_tests
    try:
        if len(profile_cases) != total_case_count:
            raise AssertionError(
                f"profile case count mismatch: expected={total_case_count}, actual={len(profile_cases)}"
            )
        if (
            len(small_results) != total_case_count
            or len(fused_results) != total_case_count
        ):
            raise AssertionError(
                "profile result count mismatch: "
                f"expected={total_case_count}, small={len(small_results)}, fused={len(fused_results)}"
            )

        for input_name in ("x", "expert_ids", "expert_scales"):
            assert_unique_tensor_storage(
                [case["fused_inputs"][input_name] for case in profile_cases],
                f"random profile fused input {input_name}",
            )
            assert_unique_tensor_storage(
                [case["small_inputs"][input_name] for case in profile_cases],
                f"random profile small-op input {input_name}",
            )
        assert_unique_tensor_storage(
            [result[0] for result in small_results], "random profile small-op output"
        )
        assert_unique_tensor_storage(
            [result[0] for result in fused_results], "random profile fused output"
        )
        assert_unique_tensor_storage(
            [result[1] for result in small_results], "random profile small-op counts"
        )
        assert_unique_tensor_storage(
            [result[1] for result in fused_results], "random profile fused counts"
        )

        if profile_cases[0].get("fixed_values", False) and total_case_count > 1:
            baseline_small = small_results[0][0][:local_num_tokens].float()
            baseline_fused = fused_results[0][0][:local_num_tokens].float()
            baseline_small_counts = small_results[0][1]
            baseline_fused_counts = fused_results[0][1]
            fixed_cross_mismatches = []
            for case_idx in range(1, total_case_count):
                small_value_error = None
                fused_value_error = None
                small_count_error = None
                fused_count_error = None
                current_small = small_results[case_idx][0][:local_num_tokens].float()
                current_fused = fused_results[case_idx][0][:local_num_tokens].float()
                try:
                    torch.testing.assert_close(
                        current_small, baseline_small, atol=0.0, rtol=0.0
                    )
                except Exception as exc:
                    small_value_error = str(exc).splitlines()[0]
                try:
                    torch.testing.assert_close(
                        current_fused, baseline_fused, atol=0.0, rtol=0.0
                    )
                except Exception as exc:
                    fused_value_error = str(exc).splitlines()[0]
                try:
                    torch.testing.assert_close(
                        small_results[case_idx][1], baseline_small_counts
                    )
                except Exception as exc:
                    small_count_error = str(exc).splitlines()[0]
                try:
                    torch.testing.assert_close(
                        fused_results[case_idx][1], baseline_fused_counts
                    )
                except Exception as exc:
                    fused_count_error = str(exc).splitlines()[0]
                if any(
                    (
                        small_value_error,
                        fused_value_error,
                        small_count_error,
                        fused_count_error,
                    )
                ):
                    small_avg, small_max, _ = summarize_output_diff(
                        baseline_small, current_small
                    )
                    fused_avg, fused_max, _ = summarize_output_diff(
                        baseline_fused, current_fused
                    )
                    fixed_cross_mismatches.append(
                        "iteration="
                        f"{case_idx}, seed={profile_cases[case_idx]['seed']}; "
                        f"small(avg={small_avg:.6f}, max={small_max:.6f}, error={small_value_error!r}), "
                        f"fused(avg={fused_avg:.6f}, max={fused_max:.6f}, error={fused_value_error!r}), "
                        f"small_counts={small_count_error!r}, fused_counts={fused_count_error!r}"
                    )
            if fixed_cross_mismatches:
                raise AssertionError(
                    "fixed-value cross-iteration mismatches: "
                    + " | ".join(fixed_cross_mismatches)
                )

        first_formal_small = None
        first_formal_fused = None
        first_formal_counts = None
        first_formal_fused_counts = None
        for test_idx in range(args.num_tests):
            result_idx = args.num_warmups + test_idx
            case = profile_cases[result_idx]
            small_output, small_counts = small_results[result_idx]
            fused_output, fused_counts = fused_results[result_idx]
            valid_small_output = small_output[:local_num_tokens]
            valid_fused_output = fused_output[:local_num_tokens]
            small_nan_count = torch.isnan(small_output).sum().item()
            fused_nan_count = torch.isnan(fused_output).sum().item()
            avg_diff, max_diff, cosine_diff = summarize_output_diff(
                valid_small_output, valid_fused_output
            )
            same_round_error = None

            small_cross_error = None
            fused_cross_error = None
            counts_cross_error = None
            fused_counts_cross_error = None
            if case.get("fixed_values", False):
                if first_formal_small is None:
                    first_formal_small = valid_small_output.float().clone()
                    first_formal_fused = valid_fused_output.float().clone()
                    first_formal_counts = small_counts.clone()
                    first_formal_fused_counts = fused_counts.clone()
                else:
                    try:
                        torch.testing.assert_close(
                            valid_small_output.float(),
                            first_formal_small,
                            atol=0.0,
                            rtol=0.0,
                        )
                    except Exception as exc:
                        small_cross_error = str(exc).splitlines()[0]
                    try:
                        torch.testing.assert_close(
                            valid_fused_output.float(),
                            first_formal_fused,
                            atol=0.0,
                            rtol=0.0,
                        )
                    except Exception as exc:
                        fused_cross_error = str(exc).splitlines()[0]
                    try:
                        torch.testing.assert_close(small_counts, first_formal_counts)
                    except Exception as exc:
                        counts_cross_error = str(exc).splitlines()[0]
                    try:
                        torch.testing.assert_close(
                            fused_counts, first_formal_fused_counts
                        )
                    except Exception as exc:
                        fused_counts_cross_error = str(exc).splitlines()[0]

            try:
                assert small_output.shape == (padded_num_tokens, args.hidden)
                assert fused_output.shape == (local_num_tokens, args.hidden)
                assert small_output.dtype == torch.bfloat16
                assert fused_output.dtype == torch.bfloat16
                assert small_counts.shape == (local_experts,)
                assert fused_counts.shape == (local_experts,)
                assert small_nan_count == 0
                assert fused_nan_count == 0
                torch.testing.assert_close(small_counts, fused_counts)
                if local_num_tokens > 0:
                    try:
                        torch.testing.assert_close(
                            valid_small_output.float(),
                            valid_fused_output.float(),
                            atol=ACCURACY_ATOL,
                            rtol=ACCURACY_RTOL,
                        )
                    except Exception as exc:
                        same_round_error = str(exc).splitlines()[0]
                        raise AssertionError(
                            f"small_vs_fused mismatch: {same_round_error}"
                        ) from exc
                if (
                    small_cross_error
                    or fused_cross_error
                    or counts_cross_error
                    or fused_counts_cross_error
                ):
                    raise AssertionError(
                        "fixed-value cross-iteration mismatch: "
                        f"small={small_cross_error!r}, fused={fused_cross_error!r}, "
                        f"small_counts={counts_cross_error!r}, "
                        f"fused_counts={fused_counts_cross_error!r}"
                    )
            except Exception as exc:
                local_error = (
                    f"[rank {rank}] profile accuracy failed: "
                    f"performance_iteration={test_idx}, result_index={result_idx}, "
                    f"seed={case['seed']}, local_num_tokens={local_num_tokens}, "
                    f"avg_diff={avg_diff:.6f}, max_diff={max_diff:.6f}, "
                    f"calc_diff={cosine_diff:.6f}, small_nan_count={small_nan_count}, "
                    f"fused_nan_count={fused_nan_count}, "
                    f"expert_ids={case['fused_inputs']['expert_ids'].cpu().tolist()}, "
                    f"small_counts={small_counts.cpu().tolist()}, "
                    f"fused_counts={fused_counts.cpu().tolist()}, "
                    f"comparison={'small_vs_fused' if same_round_error else 'profile'}: {exc}"
                )
                if (
                    small_cross_error
                    or fused_cross_error
                    or counts_cross_error
                    or fused_counts_cross_error
                ):
                    local_error += (
                        f", fixed_cross_iteration={{small={small_cross_error!r}, "
                        f"fused={fused_cross_error!r}, "
                        f"small_counts={counts_cross_error!r}, "
                        f"fused_counts={fused_counts_cross_error!r}}}"
                    )
                break
    except Exception as exc:
        local_error = f"[rank {rank}] random profile validation setup failed: {exc}"

    failure_flag = torch.tensor(
        [1 if local_error is not None else 0], dtype=torch.int32, device="npu"
    )
    dist.all_reduce(failure_flag, op=dist.ReduceOp.SUM)
    if failure_flag.item() != 0:
        if local_error is not None:
            print(local_error, flush=True)
            print_profile_case_head_bytes(
                profile_cases, small_results, fused_results, rank
            )
        dist.barrier()
        raise AssertionError(
            f"Random profile accuracy validation failed on {failure_flag.item()} of {world_size} ranks"
        )
    if rank == 0:
        print(
            f"Random profile accuracy check passed for {args.num_tests} performance iterations.",
            flush=True,
        )


def run_small_op_baseline(
    inputs: Dict[str, torch.Tensor],
    hcomm_name: str,
    rank: int,
    world_size: int,
    global_bs: int,
    args: argparse.Namespace,
    gmm_burn_in_repeats: int = 1,
    warmup_burn_in_buffers: Tuple[torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    quant_cfg = get_mx_quant_config(args)
    output_dtype = inputs["x"].dtype

    outputs = torch_npu.npu_moe_distribute_dispatch_v2(
        x=inputs["x"],
        expert_ids=inputs["expert_ids"],
        expert_scales=inputs["expert_scales"],
        scales=None,
        x_active_mask=inputs["x_active_mask"],
        group_ep=hcomm_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=args.num_experts,
        group_tp="",
        tp_world_size=1,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=quant_cfg["dispatch_quant_mode"],
        global_bs=global_bs,
        expert_token_nums_type=1,
        y_dtype=quant_cfg["dispatch_y_dtype"],
    )
    (
        expand_x,
        dynamic_scales,
        assist_info_for_combine,
        expert_token_nums,
        ep_send_counts,
        tp_send_counts,
        expand_scales,
    ) = outputs

    if args.log_quant_dtypes and not getattr(args, "_small_quant_dtype_logged", False):
        log_quant_tensor(rank, True, "dispatch.expand_x", expand_x)
        log_quant_tensor(rank, True, "dispatch.dynamic_scales_raw", dynamic_scales)
        log_quant_tensor(rank, True, "gmm1_weight_q", inputs["gmm1_weight_q"])
        log_quant_tensor(rank, True, "gmm1_weight_scale", inputs["gmm1_weight_scale"])
        log_tensor_meta(rank, True, "small_op.gmm1_weight_q", inputs["gmm1_weight_q"])

    if gmm_burn_in_repeats > 1 and warmup_burn_in_buffers is not None:
        # This burn-in only runs on the first profiler warmup iteration.
        # It does not change the routed output; it only inserts extra device
        # work before the normal small-op chain of that warmup iteration.
        burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
        for _ in range(gmm_burn_in_repeats):
            _ = torch.matmul(burn_in_lhs, burn_in_rhs)

    dynamic_scales = dynamic_scales.view(*(dynamic_scales.shape[:-1]), -1, 2).view(
        torch.float8_e8m0fnu
    )
    log_quant_tensor(
        rank, args.log_quant_dtypes, "dispatch.dynamic_scales", dynamic_scales
    )

    y1_fp = torch_npu.npu_grouped_matmul(
        x=[expand_x],
        weight=[inputs["gmm1_weight_q"]],
        scale=[inputs["gmm1_weight_scale"]],
        per_token_scale=[dynamic_scales],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_nums,
        output_dtype=output_dtype,
    )[0]
    log_quant_tensor(rank, args.log_quant_dtypes, "gmm1.output", y1_fp)
    swiglu_out = torch_npu.npu_swiglu(y1_fp)
    log_quant_tensor(rank, args.log_quant_dtypes, "swiglu.output", swiglu_out)

    x2, x2_scale = torch_npu.npu_dynamic_mx_quant(
        swiglu_out, dst_type=quant_cfg["quant_dst_type"]
    )
    x2 = x2.view(quant_cfg["origin_dtype"])

    x2_scale = x2_scale.view(torch.float8_e8m0fnu)
    log_quant_tensor(rank, args.log_quant_dtypes, "requant.x2", x2)
    log_quant_tensor(rank, args.log_quant_dtypes, "requant.x2_scale", x2_scale)
    if args.log_quant_dtypes and not getattr(
        args, "_small_gmm2_weight_meta_logged", False
    ):
        log_tensor_meta(rank, True, "small_op.gmm2_weight_q", inputs["gmm2_weight_q"])
        args._small_gmm2_weight_meta_logged = True
    y2_fp = torch_npu.npu_grouped_matmul(
        x=[x2],
        weight=[inputs["gmm2_weight_q"]],
        scale=[inputs["gmm2_weight_scale"]],
        per_token_scale=[x2_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_nums,
        output_dtype=output_dtype,
    )[0]
    log_quant_tensor(rank, args.log_quant_dtypes, "gmm2.output", y2_fp)
    if args.log_quant_dtypes:
        args._small_quant_dtype_logged = True
    output = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=y2_fp,
        expert_ids=inputs["expert_ids"],
        assist_info_for_combine=assist_info_for_combine,
        ep_send_counts=ep_send_counts,
        expert_scales=inputs["expert_scales"],
        x_active_mask=inputs["x_active_mask"],
        group_ep=hcomm_name,
        ep_world_size=world_size,
        ep_rank_id=rank,
        moe_expert_num=args.num_experts,
        tp_send_counts=tp_send_counts,
        expand_scales=expand_scales,
        group_tp="",
        tp_world_size=1,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        global_bs=global_bs,
    )
    return output, expert_token_nums.to(torch.int32)


def run_buffer_fused(
    buffer: deep_ep.Buffer,
    inputs: Dict[str, torch.Tensor],
    num_max_dispatch_tokens_per_rank: int,
    args: argparse.Namespace,
    kernel_trace_dir: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    log_tensor_meta(
        rank, args.log_quant_dtypes, "gmm1_weight_q", inputs["gmm1_weight_q"]
    )
    log_tensor_meta(
        rank, args.log_quant_dtypes, "gmm2_weight_q", inputs["gmm2_weight_q"]
    )

    output, ep_recv_count = buffer.fused_deep_moe(
        inputs["x"],
        inputs["expert_ids"],
        inputs["expert_scales"],
        inputs["gmm1_weight_q"],
        inputs["gmm1_weight_scale"],
        inputs["gmm2_weight_q"],
        inputs["gmm2_weight_scale"],
        num_max_dispatch_tokens_per_rank,
        args.num_experts,
        FUSED_COMPAT_QUANT_MODE,
        profile_enable=kernel_trace_dir is not None,
    )
    return output, ep_recv_count


def run_buffer_fused_with_burn_in(
    buffer: deep_ep.Buffer,
    inputs: Dict[str, torch.Tensor],
    num_max_dispatch_tokens_per_rank: int,
    args: argparse.Namespace,
    fused_burn_in_repeats: int = 1,
    warmup_burn_in_buffers: Tuple[torch.Tensor, torch.Tensor] = None,
    kernel_trace_dir: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The extra burn-in is only used for the first profiler warmup iteration.
    # It intentionally stays on the same execution path as the fused op so the
    # host has more time to enqueue the remaining fused iterations while this
    # warmup is still occupying device time.
    if fused_burn_in_repeats > 1 and warmup_burn_in_buffers is not None:
        burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
        for _ in range(fused_burn_in_repeats):
            _ = torch.matmul(burn_in_lhs, burn_in_rhs)

    return run_buffer_fused(
        buffer,
        inputs,
        num_max_dispatch_tokens_per_rank,
        args,
        kernel_trace_dir=kernel_trace_dir,
    )


def format_triplet(name: str, values_us: Tuple[float, float, float]) -> str:
    avg_us, min_us, max_us = values_us
    return f"{name}: avg={avg_us:.2f} us, min={min_us:.2f} us, max={max_us:.2f} us"


def summarize_profile_durations(durations_s) -> Tuple[float, float, float]:
    return (
        float(durations_s.mean()),
        float(durations_s.min()),
        float(durations_s.max()),
    )


def summarize_profile_breakdown(step_durations_s) -> List[Tuple[float, float, float]]:
    return [
        (
            float(step_durations_s[:, idx].mean()),
            float(step_durations_s[:, idx].min()),
            float(step_durations_s[:, idx].max()),
        )
        for idx in range(step_durations_s.shape[1])
    ]


def print_profile_match_debug(
    rank: int, title: str, discovered_names, matched_sequences
):
    if rank != 0:
        return
    if matched_sequences:
        print(f"{title} matched profiler event sequences:", flush=True)
        for idx, sequence in enumerate(matched_sequences, start=1):
            print(f"  [{idx}] {' -> '.join(sequence)}", flush=True)
    else:
        print(f"{title} discovered profiler events:", flush=True)
        for name in discovered_names:
            print(f"  - {name}", flush=True)


def print_profile_iteration_debug(rank: int, title: str, debug_info):
    if rank != 0:
        return
    print(
        f"{title} profiler iterations: "
        f"matched_total_iterations={debug_info['matched_total_iterations']}, "
        f"dropped_warmup_iterations={debug_info['dropped_warmup_iterations']}, "
        f"counted_iterations={debug_info['counted_iterations']}",
        flush=True,
    )


def print_rank_skew_summary(gathered_small, gathered_small_breakdown, gathered_fused):
    # This table is primarily a maintenance/debugging view. It keeps the
    # per-rank stage averages together so rank-to-rank skew can be judged
    # without manually correlating multiple profiler tables or traces.
    small_tensor = torch.stack(gathered_small).cpu()
    fused_tensor = (
        torch.stack(gathered_fused).cpu() if gathered_fused is not None else None
    )
    breakdown_tensor = torch.stack(gathered_small_breakdown).cpu()

    print("rank skew summary:", flush=True)
    print(
        "| Rank | Dispatch (us) | GMM1 (us) | SwiGLU (us) | Requant (us) | GMM2 (us) | Combine (us) | Small Total (us) | Fused (us) |",
        flush=True,
    )
    print(
        "|-----:|--------------:|----------:|------------:|-------------:|----------:|-------------:|-----------------:|-----------:|",
        flush=True,
    )

    for rank_idx in range(len(gathered_small)):
        breakdown_avg = breakdown_tensor[rank_idx, :, 0]
        dispatch_avg = float(breakdown_avg[0].item())
        gmm1_avg = float(breakdown_avg[1].item())
        swiglu_avg = float(breakdown_avg[2].item())
        requant_avg = float(breakdown_avg[3].item())
        gmm2_avg = float(breakdown_avg[4].item())
        combine_avg = float(breakdown_avg[5].item())
        small_avg = float(small_tensor[rank_idx, 0].item())
        fused_avg = (
            float(fused_tensor[rank_idx, 0].item())
            if fused_tensor is not None
            else float("nan")
        )
        fused_text = f"{fused_avg * 1e6:.2f}" if fused_tensor is not None else "-"
        print(
            f"| {rank_idx:>4} | {dispatch_avg * 1e6:>13.2f} | {gmm1_avg * 1e6:>9.2f} | "
            f"{swiglu_avg * 1e6:>11.2f} | {requant_avg * 1e6:>12.2f} | {gmm2_avg * 1e6:>9.2f} | "
            f"{combine_avg * 1e6:>12.2f} | {small_avg * 1e6:>16.2f} | {fused_text:>10} |",
            flush=True,
        )

    mean_breakdown_avg = breakdown_tensor[:, :, 0].mean(dim=0)
    mean_small_avg = float(small_tensor[:, 0].mean().item())
    mean_fused_avg = (
        float(fused_tensor[:, 0].mean().item())
        if fused_tensor is not None
        else float("nan")
    )
    mean_fused_text = f"{mean_fused_avg * 1e6:.2f}" if fused_tensor is not None else "-"
    print(
        f"| {'mean':>4} | {float(mean_breakdown_avg[0].item()) * 1e6:>13.2f} | "
        f"{float(mean_breakdown_avg[1].item()) * 1e6:>9.2f} | "
        f"{float(mean_breakdown_avg[2].item()) * 1e6:>11.2f} | "
        f"{float(mean_breakdown_avg[3].item()) * 1e6:>12.2f} | "
        f"{float(mean_breakdown_avg[4].item()) * 1e6:>9.2f} | "
        f"{float(mean_breakdown_avg[5].item()) * 1e6:>12.2f} | "
        f"{mean_small_avg * 1e6:>16.2f} | {mean_fused_text:>10} |",
        flush=True,
    )


def print_comm_overlap_rate(total_small_stats, total_fused_stats, mean_breakdown_stats):
    # Communication overlap rate is reported as a coarse derived metric using
    # mean-over-ranks averages:
    #   (small_total - fused_total) / (dispatch + combine)
    # It is intended for cross-run trend comparison rather than as a strict
    # micro-kernel efficiency metric.
    dispatch_avg = mean_breakdown_stats[0][0]
    combine_avg = mean_breakdown_stats[5][0]
    denom = dispatch_avg + combine_avg
    if denom <= 0:
        print("communication overlap rate: N/A (dispatch + combine <= 0)", flush=True)
        return
    overlap_rate = (total_small_stats[0] - total_fused_stats[0]) / denom
    print(
        "communication overlap rate: "
        f"(({total_small_stats[0] * 1e6:.2f} - {total_fused_stats[0] * 1e6:.2f}) / "
        f"({dispatch_avg * 1e6:.2f} + {combine_avg * 1e6:.2f})) = {overlap_rate:.4f}",
        flush=True,
    )


def build_trace_path(trace_dir: str, rank: int, tag: str) -> str:
    os.makedirs(trace_dir, exist_ok=True)
    return os.path.join(trace_dir, f"{tag}_rank{rank}.json")


def format_stats_row(
    name: str, stats_s: Tuple[float, float, float], ratio_pct: float = None
) -> str:
    avg_us, min_us, max_us = (value * 1e6 for value in stats_s)
    ratio_str = "-" if ratio_pct is None else f"{ratio_pct:6.2f}%"
    return f"| {name:<12} | {avg_us:>12.2f} | {min_us:>12.2f} | {max_us:>12.2f} | {ratio_str:>8} |"


def print_stats_table(
    title: str, rows: List[Tuple[str, Tuple[float, float, float], float]]
):
    print(title, flush=True)
    print(
        "| Stage        |    Avg (us) |    Min (us) |    Max (us) |  Share % |",
        flush=True,
    )
    print(
        "|--------------|-------------:|-------------:|-------------:|---------:|",
        flush=True,
    )
    for name, stats_s, ratio_pct in rows:
        print(format_stats_row(name, stats_s, ratio_pct), flush=True)


def build_small_rows(
    total_stats: Tuple[float, float, float],
    breakdown_stats: List[Tuple[float, float, float]],
) -> List[Tuple[str, Tuple[float, float, float], float]]:
    rows = []
    for label, stats_tuple in zip(SMALL_OP_EVENT_LABELS, breakdown_stats):
        ratio_pct = (
            stats_tuple[0] / total_stats[0] * 100.0 if total_stats[0] > 0 else 0.0
        )
        rows.append((label, stats_tuple, ratio_pct))
    rows.append(("total", total_stats, 100.0))
    return rows


def print_per_rank_profile_tables(
    gathered_small,
    gathered_small_breakdown,
    gathered_fused,
):
    world_size = 0
    if gathered_small is not None:
        world_size = len(gathered_small)
    elif gathered_fused is not None:
        world_size = len(gathered_fused)

    for rank_idx in range(world_size):
        small_stats = None
        fused_stats = None

        if gathered_small is not None:
            small_stats = tuple(
                float(v) for v in gathered_small[rank_idx].cpu().tolist()
            )
            small_breakdown_stats = [
                tuple(float(v) for v in stats_values)
                for stats_values in gathered_small_breakdown[rank_idx].cpu().tolist()
            ]
            print_stats_table(
                f"small-op breakdown (rank {rank_idx}):",
                build_small_rows(small_stats, small_breakdown_stats),
            )

        if gathered_fused is not None:
            fused_stats = tuple(
                float(v) for v in gathered_fused[rank_idx].cpu().tolist()
            )
            print_stats_table(
                f"fused buffer path (rank {rank_idx}):",
                [("fused", fused_stats, 100.0)],
            )

        if small_stats is not None and fused_stats is not None:
            speedup = small_stats[0] / fused_stats[0]
            delta_pct = (fused_stats[0] - small_stats[0]) / small_stats[0] * 100.0
            print(
                f"[rank {rank_idx}] small_total_avg_us={small_stats[0] * 1e6:.2f}, "
                f"fused_avg_us={fused_stats[0] * 1e6:.2f}, "
                f"speedup={speedup:.4f}x, delta_pct={delta_pct:.2f}%",
                flush=True,
            )


def print_fused_counts_table(gathered_fused_counts: List[torch.Tensor]):
    print("fused counts (per rank):", flush=True)
    print("| Rank | Local Experts | Fused Counts         | Sum |", flush=True)
    print("|-----:|--------------:|----------------------|----:|", flush=True)
    for rank_idx, counts in enumerate(gathered_fused_counts):
        counts_list = [int(v) for v in counts.cpu().tolist()]
        print(
            f"| {rank_idx:>4} | {len(counts_list):>13} | {str(counts_list):<20} | {sum(counts_list):>3} |",
            flush=True,
        )


def summarize_output_diff(
    reference: torch.Tensor, actual: torch.Tensor
) -> Tuple[float, float, float]:
    if reference.numel() == 0:
        return 0.0, 0.0, 0.0
    reference_f = reference.float()
    actual_f = actual.float()
    eps = 1e-8
    reference_nozero = torch.where(reference_f == 0, eps, reference_f)
    rel_diff = torch.abs(actual_f - reference_f) / torch.abs(reference_nozero)
    avg_diff = torch.mean(rel_diff).item()
    max_diff = torch.max(rel_diff).item()
    cosine_diff = calc_diff(reference_f, actual_f)
    return avg_diff, max_diff, cosine_diff


def summarize_tensor_stats(tensor: torch.Tensor) -> Tuple[float, float]:
    if tensor.numel() == 0:
        return 0.0, 0.0
    tensor_f = tensor.float()
    return tensor_f.abs().max().item(), tensor_f.mean().item()


def get_uniform_expected_counts(
    total_num_tokens: int,
    num_topk: int,
    num_experts: int,
    local_experts: int,
):
    # The synthetic expert_ids in this test are built with
    # arange(...).remainder(num_experts), so exact per-expert expected counts
    # only exist when the total routed assignments divide evenly by num_experts.
    # When that is not true, we intentionally skip strict expected_counts
    # assertions and only enforce small_counts == fused_counts.
    total_assignments = total_num_tokens * num_topk
    if total_assignments % num_experts != 0:
        return None, total_assignments
    expected_per_expert = total_assignments // num_experts
    expected_counts = torch.full(
        (local_experts,),
        expected_per_expert,
        dtype=torch.int32,
        device="npu",
    )
    return expected_counts, total_assignments


@torch.inference_mode()
def run_rank(local_rank: int, num_processes: int, args: argparse.Namespace):
    group = None
    group_small = None
    group_fused = None
    pending_exc = None
    pending_tb = None
    try:
        rank, world_size, group = init_dist(local_rank, num_processes)
        torch.manual_seed(2026 + rank)
        ranks = list(range(world_size))
        group_fused = dist.new_group(ranks)
        group_small = dist.new_group(ranks)
        buffer = deep_ep.Buffer(group_fused, low_latency_mode=True)

        if args.single_active_rank is not None:
            token_delta = 0
            local_num_tokens = 1 if rank == args.single_active_rank else 0
        else:
            if args.num_token_jitter > 0:
                token_jitter_rng = random.Random(args.num_token_jitter_seed + rank)
                token_delta = token_jitter_rng.randint(
                    -args.num_token_jitter, args.num_token_jitter
                )
            else:
                token_delta = 0
            local_num_tokens = max(1, min(256, args.num_tokens + token_delta))
        local_num_tokens_tensor = torch.tensor(
            [local_num_tokens], dtype=torch.int64, device="npu"
        )
        gathered_num_tokens = [
            torch.empty_like(local_num_tokens_tensor) for _ in range(world_size)
        ]
        dist.all_gather(gathered_num_tokens, local_num_tokens_tensor)
        all_num_tokens = [int(t.item()) for t in gathered_num_tokens]
        real_global_bs = sum(all_num_tokens)
        padded_num_tokens = max(all_num_tokens)
        small_op_global_bs = padded_num_tokens * world_size
        token_prefix_before_rank = sum(all_num_tokens[:rank])
        has_valid_tokens = local_num_tokens > 0

        inputs = make_umdk_static_inputs(
            rank,
            world_size,
            local_num_tokens,
            token_prefix_before_rank,
            args,
        )
        small_op_inputs = make_small_op_padded_inputs(
            inputs, local_num_tokens, padded_num_tokens
        )
        warmup_burn_in_buffers = make_small_warmup_burn_in_buffers(
            local_num_tokens, args
        )
        hcomm_name_small = group_small._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(rank)
        local_experts = args.num_experts // world_size
        if args.single_active_rank is not None:
            expected_counts, total_assignments = None, real_global_bs * args.num_topk
        else:
            expected_counts, total_assignments = get_uniform_expected_counts(
                real_global_bs,
                args.num_topk,
                args.num_experts,
                local_experts,
            )
        local_topk_tensor = torch.full(
            (args.num_topk,),
            -1,
            dtype=torch.int32,
            device="npu",
        )
        if args.single_active_rank is not None and has_valid_tokens:
            local_topk_tensor.copy_(inputs["expert_ids"][0])
        gathered_topk = [torch.empty_like(local_topk_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_topk, local_topk_tensor)
        if rank == 0:
            print(
                "Config: "
                f"num_processes={num_processes}, "
                f"base_num_tokens={args.num_tokens}, "
                f"single_active_rank={args.single_active_rank}, "
                f"num_token_jitter={args.num_token_jitter}, "
                f"num_token_jitter_seed={args.num_token_jitter_seed}, "
                f"per_rank_num_tokens={all_num_tokens}, "
                f"padded_num_tokens={padded_num_tokens}, "
                f"real_global_bs={real_global_bs}, "
                f"small_op_global_bs={small_op_global_bs}, "
                f"hidden={args.hidden}, "
                f"moe_intermediate_size={args.moe_intermediate_size}, "
                f"num_experts={args.num_experts}, "
                f"num_topk={args.num_topk}, "
                f"quant={args.quant}, "
                f"weight_format={args.weight_format}, "
                f"kernel_trace_dir={args.kernel_trace_dir}, "
                f"num_warmups={args.num_warmups}, "
                f"num_tests={args.num_tests}, "
                f"small_first_warmup_gmm_burn_in_repeats={SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS}",
                flush=True,
            )
            if args.single_active_rank is not None:
                inactive_ranks = [
                    idx for idx in range(world_size) if idx != args.single_active_rank
                ]
                print(
                    "Special case: "
                    f"single_active_rank={args.single_active_rank}, "
                    f"active_rank={args.single_active_rank}, "
                    f"inactive_ranks={inactive_ranks}",
                    flush=True,
                )
                active_topk_experts = [
                    int(v)
                    for v in gathered_topk[args.single_active_rank].cpu().tolist()
                ]
                active_topk_ranks = [
                    expert_id // local_experts for expert_id in active_topk_experts
                ]
                print(
                    "Randomized single-active topk: "
                    f"seed={args.single_active_topk_seed}, "
                    f"active_token_topk_experts={active_topk_experts}, "
                    f"active_token_topk_ranks={active_topk_ranks}",
                    flush=True,
                )
                print(
                    "Strict expected-counts check is skipped in single_active_rank mode "
                    "because topk experts are randomized across the global expert space.",
                    flush=True,
                )
            elif expected_counts is None:
                print(
                    "Warning: num_tokens * num_processes * num_topk is not divisible by num_experts. "
                    "This synthetic expert_ids construction will produce non-uniform expert token counts, "
                    "so strict per-expert expected_counts checks are skipped; only small_counts == fused_counts "
                    "is enforced.",
                    flush=True,
                )

        small_output, small_counts = run_small_op_baseline(
            small_op_inputs,
            hcomm_name_small,
            rank,
            world_size,
            small_op_global_bs,
            args,
        )
        fused_output, fused_counts = run_buffer_fused(
            buffer, inputs, padded_num_tokens, args
        )
        torch.npu.synchronize()

        assert small_output.shape == (padded_num_tokens, args.hidden)
        assert fused_output.shape == (local_num_tokens, args.hidden)
        assert small_output.dtype == torch.bfloat16
        assert fused_output.dtype == torch.bfloat16
        assert small_counts.shape == (local_experts,)
        assert fused_counts.shape == (local_experts,)
        assert torch.isnan(small_output).sum().item() == 0
        assert torch.isnan(fused_output).sum().item() == 0

        torch.testing.assert_close(small_counts, fused_counts)
        if expected_counts is not None:
            torch.testing.assert_close(small_counts, expected_counts)
            torch.testing.assert_close(fused_counts, expected_counts)
        valid_token_num = local_num_tokens

        avg_diff, max_diff, cosine_diff = summarize_output_diff(
            small_output[:valid_token_num], fused_output[:valid_token_num]
        )
        small_absmax, small_mean = summarize_tensor_stats(
            small_output[:valid_token_num]
        )
        fused_absmax, fused_mean = summarize_tensor_stats(
            fused_output[:valid_token_num]
        )
        diag_tensor = torch.tensor(
            [
                avg_diff,
                max_diff,
                cosine_diff,
                small_absmax,
                small_mean,
                fused_absmax,
                fused_mean,
                float(has_valid_tokens),
            ],
            dtype=torch.float32,
            device="npu",
        )
        gathered_diag = [torch.empty_like(diag_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_diag, diag_tensor)
        gathered_fused_counts = [
            torch.empty_like(fused_counts) for _ in range(world_size)
        ]
        dist.all_gather(gathered_fused_counts, fused_counts)
        dist.barrier()
        if has_valid_tokens:
            torch.testing.assert_close(
                small_output[:valid_token_num].float(),
                fused_output[:valid_token_num].float(),
                atol=ACCURACY_ATOL,
                rtol=ACCURACY_RTOL,
            )
        if rank == 0:
            print_fused_counts_table(gathered_fused_counts)
            gathered_diag_cpu = torch.stack(gathered_diag).cpu()
            skipped_accuracy_ranks = [
                idx
                for idx in range(world_size)
                if gathered_diag_cpu[idx, 7].item() == 0.0
            ]
            print(
                "Accuracy check passed. "
                f"avg_diff={gathered_diag_cpu[:, 0].max().item():.6f}, "
                f"max_diff={gathered_diag_cpu[:, 1].max().item():.6f}, "
                f"calc_diff={gathered_diag_cpu[:, 2].max().item():.6f}",
                flush=True,
            )
            if skipped_accuracy_ranks:
                print(
                    "Skipped per-token accuracy comparison for zero-token ranks: "
                    f"active_rank={args.single_active_rank}, "
                    f"skipped_ranks={skipped_accuracy_ranks}",
                    flush=True,
                )

        dist.barrier()
        small_stats = None
        small_breakdown_stats = None
        fused_stats = None
        profile_cases = None
        small_profile_results = []
        fused_profile_results = []
        if args.random_profile_cases or args.fixed_profile_cases:
            case_builder = (
                make_fixed_profile_cases
                if args.fixed_profile_cases
                else make_random_profile_cases
            )
            profile_cases = case_builder(
                inputs,
                rank,
                world_size,
                local_num_tokens,
                padded_num_tokens,
                args,
            )
            # Keep random input construction outside both profiler regions.
            torch.npu.synchronize()
            if rank == 0:
                print(
                    f"{'Fixed' if args.fixed_profile_cases else 'Random'} profile cases prepared: "
                    f"count={len(profile_cases)}, seed={args.random_profile_seed}",
                    flush=True,
                )

        if True:
            small_trace_path = (
                build_trace_path(args.trace_dir, rank, "small_op")
                if args.trace_dir is not None
                else None
            )
            small_profile_call_idx = 0

            def small_profile_fn():
                nonlocal small_profile_call_idx
                # Only the first warmup iteration gets the extra burn-in.
                # All later warmups and all counted iterations run the normal
                # small-op path so the reported profiler statistics remain
                # comparable to real steady-state execution.
                gmm_burn_in_repeats = (
                    SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS
                    if small_profile_call_idx == 0 and args.num_warmups > 0
                    else 1
                )
                profile_inputs = (
                    profile_cases[small_profile_call_idx]["small_inputs"]
                    if profile_cases is not None
                    else small_op_inputs
                )
                small_profile_call_idx += 1
                result = run_small_op_baseline(
                    profile_inputs,
                    hcomm_name_small,
                    rank,
                    world_size,
                    small_op_global_bs,
                    args,
                    gmm_burn_in_repeats=gmm_burn_in_repeats,
                    warmup_burn_in_buffers=warmup_burn_in_buffers,
                )
                if profile_cases is not None:
                    small_profile_results.append(result)
                return result

            (
                small_durations,
                small_event_names,
                small_matched_sequences,
                small_step_durations,
                small_debug_info,
            ) = profile_npu_event_sequences(
                small_profile_fn,
                SMALL_OP_EVENT_PATTERNS,
                num_warmups=args.num_warmups,
                num_tests=args.num_tests,
                suppress_kineto_output=True,
                trace_path=small_trace_path,
                allow_no_match=args.dump_profile_events,
            )
            if args.dump_profile_events:
                print_profile_iteration_debug(rank, "small-op", small_debug_info)
                print_profile_match_debug(
                    rank, "small-op", small_event_names, small_matched_sequences
                )
            if len(small_durations) == 0:
                raise AssertionError(
                    "No matched NPU event sequence found for small-op. "
                    f"Patterns={SMALL_OP_EVENT_PATTERNS}. "
                    f"Discovered events={small_event_names}"
                )
            small_stats = summarize_profile_durations(small_durations)
            small_breakdown_stats = summarize_profile_breakdown(small_step_durations)
        dist.barrier()

        if True:
            fused_trace_path = (
                build_trace_path(args.trace_dir, rank, "fused")
                if args.trace_dir is not None
                else None
            )
            fused_profile_call_idx = 0

            if args.kernel_trace_dir is not None:
                print(
                    f"[rank{rank}] begin fused kernel profiling: "
                    f"trace_dir={args.kernel_trace_dir}, warmups={args.num_warmups}, tests={args.num_tests}",
                    flush=True,
                )
                buffer.runtime.begin_profile(
                    args.num_warmups, args.num_tests, args.kernel_trace_dir
                )

            def fused_profile_fn():
                nonlocal fused_profile_call_idx
                # Insert extra device work only between the first and second
                # fused warmup iterations. This keeps warmup #0 on the normal
                # fused path while giving the host more time before warmup #1.
                if (
                    fused_profile_call_idx == 1
                    and args.num_warmups >= 2
                    and warmup_burn_in_buffers is not None
                ):
                    burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
                    for _ in range(SMALL_FIRST_WARMUP_GMM_BURN_IN_REPEATS):
                        _ = torch.matmul(burn_in_lhs, burn_in_rhs)
                profile_inputs = (
                    profile_cases[fused_profile_call_idx]["fused_inputs"]
                    if profile_cases is not None
                    else inputs
                )
                fused_profile_call_idx += 1
                result = run_buffer_fused(
                    buffer,
                    profile_inputs,
                    padded_num_tokens,
                    args,
                    kernel_trace_dir=args.kernel_trace_dir,
                )
                if profile_cases is not None:
                    fused_profile_results.append(result)
                return result

            try:
                (
                    fused_durations,
                    fused_event_names,
                    fused_matched_sequences,
                    _,
                    fused_debug_info,
                ) = profile_npu_event_sequences(
                    fused_profile_fn,
                    FUSED_EVENT_PATTERNS,
                    num_warmups=args.num_warmups,
                    num_tests=args.num_tests,
                    suppress_kineto_output=True,
                    trace_path=fused_trace_path,
                    allow_no_match=args.dump_profile_events,
                )
                if args.dump_profile_events:
                    print_profile_iteration_debug(rank, "fused", fused_debug_info)
                    print_profile_match_debug(
                        rank, "fused", fused_event_names, fused_matched_sequences
                    )
                if len(fused_durations) == 0:
                    raise AssertionError(
                        "No matched NPU event sequence found for fused. "
                        f"Patterns={FUSED_EVENT_PATTERNS}. "
                        f"Discovered events={fused_event_names}"
                    )
                fused_stats = summarize_profile_durations(fused_durations)
            finally:
                if args.kernel_trace_dir is not None:
                    print(
                        f"[rank{rank}] end fused kernel profiling: trace_dir={args.kernel_trace_dir}",
                        flush=True,
                    )
                    buffer.runtime.end_profile()
        dist.barrier()

        if profile_cases is not None:
            validate_random_profile_results(
                profile_cases,
                small_profile_results,
                fused_profile_results,
                rank,
                world_size,
                local_num_tokens,
                padded_num_tokens,
                local_experts,
                args,
            )
            dist.barrier()

        if small_stats is not None:
            small_stats_tensor = torch.tensor(
                small_stats, dtype=torch.float32, device="npu"
            )
            gathered_small = [
                torch.empty_like(small_stats_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_small, small_stats_tensor)
            small_breakdown_tensor = torch.tensor(
                small_breakdown_stats, dtype=torch.float32, device="npu"
            )
            gathered_small_breakdown = [
                torch.empty_like(small_breakdown_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_small_breakdown, small_breakdown_tensor)
        else:
            gathered_small = None
            gathered_small_breakdown = None

        if fused_stats is not None:
            fused_stats_tensor = torch.tensor(
                fused_stats, dtype=torch.float32, device="npu"
            )
            gathered_fused = [
                torch.empty_like(fused_stats_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_fused, fused_stats_tensor)
        else:
            gathered_fused = None

        if rank == 0:
            print("Profiled NPU op time:", flush=True)
            small_mean = None
            fused_mean = None
            total_small_stats = None
            mean_breakdown_stats = None
            total_fused_stats = None
            if gathered_small is not None:
                small_mean = torch.stack(gathered_small).mean(dim=0).cpu().tolist()
                total_small_stats = tuple(small_mean)
                small_breakdown_mean = (
                    torch.stack(gathered_small_breakdown).mean(dim=0).cpu().tolist()
                )
                mean_breakdown_stats = [
                    tuple(float(v) for v in stats_values)
                    for stats_values in small_breakdown_mean
                ]
                print_stats_table(
                    "small-op breakdown (mean over ranks):",
                    build_small_rows(
                        total_small_stats,
                        mean_breakdown_stats,
                    ),
                )
            if gathered_fused is not None:
                fused_mean = torch.stack(gathered_fused).mean(dim=0).cpu().tolist()
                total_fused_stats = tuple(fused_mean)
                fused_rows = [("fused", total_fused_stats, 100.0)]
                print_stats_table("fused buffer path (mean over ranks):", fused_rows)
            if small_mean is not None and fused_mean is not None:
                speedup = small_mean[0] / fused_mean[0]
                delta_pct = (fused_mean[0] - small_mean[0]) / small_mean[0] * 100.0
                print(f"speedup={speedup:.4f}x, delta_pct={delta_pct:.2f}%", flush=True)
                print_rank_skew_summary(
                    gathered_small,
                    gathered_small_breakdown,
                    gathered_fused,
                )
                print_comm_overlap_rate(
                    total_small_stats,
                    total_fused_stats,
                    mean_breakdown_stats,
                )
            if args.single_active_rank is not None:
                active_rank = args.single_active_rank
                if gathered_small is not None:
                    active_small_stats = tuple(
                        float(v) for v in gathered_small[active_rank].cpu().tolist()
                    )
                    active_small_breakdown = [
                        tuple(float(v) for v in stats_values)
                        for stats_values in gathered_small_breakdown[active_rank]
                        .cpu()
                        .tolist()
                    ]
                    print_stats_table(
                        f"small-op breakdown (active rank {active_rank}):",
                        build_small_rows(
                            active_small_stats,
                            active_small_breakdown,
                        ),
                    )
                if gathered_fused is not None:
                    active_fused_stats = tuple(
                        float(v) for v in gathered_fused[active_rank].cpu().tolist()
                    )
                    print_stats_table(
                        f"fused buffer path (active rank {active_rank}):",
                        [("fused", active_fused_stats, 100.0)],
                    )
            if args.print_per_rank_profile:
                print_per_rank_profile_tables(
                    gathered_small,
                    gathered_small_breakdown,
                    gathered_fused,
                )
    except Exception as exc:
        pending_exc = exc
        pending_tb = traceback.format_exc()
    finally:
        if pending_tb is not None:
            rank_text = rank if "rank" in locals() else f"local_rank={local_rank}"
            print(
                f"[{rank_text}] exception captured before finalize:\n{pending_tb}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()
            if group_small is not None:
                dist.destroy_process_group(group_small)
            if group_fused is not None:
                dist.destroy_process_group(group_fused)
            dist.destroy_process_group()
    if pending_exc is not None:
        raise pending_exc.with_traceback(pending_exc.__traceback__)


def main():
    parser = argparse.ArgumentParser(
        description="A5 fused vs small-op correctness and profiler performance comparison"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of spawned ranks/devices to use for the test.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=32,
        help="Per-rank token count for the routed tokens.",
    )
    parser.add_argument(
        "--num-token-jitter",
        type=int,
        default=0,
        help="Optional per-rank symmetric integer jitter applied to --num-tokens before generating inputs.",
    )
    parser.add_argument(
        "--num-token-jitter-seed",
        type=int,
        default=2026,
        help="Base seed used to derive deterministic per-rank token jitter.",
    )
    parser.add_argument(
        "--single-active-rank",
        type=int,
        help="Optional special-case mode: only this rank uses 1 real token and every other rank uses 0 tokens.",
    )
    parser.add_argument(
        "--single-active-topk-seed",
        type=int,
        default=2026,
        help="Base seed used to generate randomized global topk experts in --single-active-rank mode.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=7168,
        help="Hidden size of the MoE input/output tensor.",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=3072,
        help="Per-expert intermediate size; 2x this value must satisfy the A5 GMM1 constraint.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=64,
        help="Global number of routed experts across all ranks.",
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=6,
        help="Top-k experts selected for each token.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=5,
        help="Number of profiler-only warmup iterations to exclude from performance statistics.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=30,
        help="Number of counted performance iterations when --profile-num-tests is not set.",
    )
    parser.add_argument(
        "--random-profile-cases",
        action="store_true",
        help=(
            "Use distinct randomized activation/routing inputs for every profiler "
            "iteration and compare results afterward."
        ),
    )
    parser.add_argument(
        "--random-profile-seed",
        type=int,
        default=2026,
        help="Base seed for deterministic per-iteration randomized profiler inputs.",
    )
    parser.add_argument(
        "--fixed-profile-cases",
        action="store_true",
        help=(
            "Use identical activation/routing values with distinct tensor storage "
            "for every profiler iteration; implies profile-case validation."
        ),
    )
    parser.add_argument(
        "--quant",
        choices=tuple(MX_QUANT_CONFIGS.keys()),
        default="fp8_e4m3",
        help="Unified MX quant dtype for the small-op chain and fused GMM weights.",
    )
    parser.add_argument(
        "--weight-format",
        choices=("ND", "NZ"),
        default="ND",
        help="Storage format for quantized GMM weights.",
    )
    parser.add_argument(
        "--trace-dir",
        help="Optional directory to export profiler chrome traces.",
    )
    parser.add_argument(
        "--kernel-trace-dir",
        help="Optional directory to export A5 fused kernel traceEvents JSON files.",
    )
    parser.add_argument(
        "--log-quant-dtypes",
        action="store_true",
        help="Print dtype/shape of key quantized tensors on rank0 during the first small-op run.",
    )
    parser.add_argument(
        "--dump-profile-events",
        "--debug",
        dest="dump_profile_events",
        action="store_true",
        help="Print matched profiler iterations and discovered event names for debugging.",
    )
    parser.add_argument(
        "--print-per-rank-profile",
        action="store_true",
        help="Print per-rank small-op/fused profiler tables in addition to the mean-over-ranks summary.",
    )
    args = parser.parse_args()

    gmm1_hidden = 2 * args.moe_intermediate_size
    if args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if args.num_experts % args.num_processes != 0:
        parser.error("--num-experts must be divisible by --num-processes")
    if args.single_active_rank is not None and not (
        0 <= args.single_active_rank < args.num_processes
    ):
        parser.error("--single-active-rank must be in [0, --num-processes - 1]")
    if not 1 <= args.num_topk <= min(args.num_experts, 12):
        parser.error("--num-topk must be in [1, min(--num-experts, 12)]")
    if not 1 <= args.num_tokens <= 256:
        parser.error("--num-tokens must be in [1, 256]")
    if args.num_token_jitter < 0:
        parser.error("--num-token-jitter must be non-negative")
    if args.single_active_rank is not None and args.num_token_jitter > 0:
        parser.error("--single-active-rank cannot be combined with --num-token-jitter")
    if not 512 <= args.hidden <= 7168:
        parser.error("--hidden must be in [512, 7168]")
    if not 1024 <= gmm1_hidden <= 6144 or gmm1_hidden % 1024 != 0:
        parser.error(
            "2 * --moe-intermediate-size must be in [1024, 6144] and divisible by 1024"
        )
    if args.num_warmups < 0:
        parser.error("--num-warmups must be non-negative")
    if args.num_tests <= 0:
        parser.error("--num-tests must be positive")
    if args.random_profile_seed < 0:
        parser.error("--random-profile-seed must be non-negative")
    if args.random_profile_cases and args.fixed_profile_cases:
        parser.error(
            "--random-profile-cases and --fixed-profile-cases are mutually exclusive"
        )
    if args.quant == "fp4_e2m1" and args.hidden % 2 != 0:
        parser.error("--hidden must be even when --quant is fp4_e2m1")

    torch.multiprocessing.spawn(
        run_rank, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
