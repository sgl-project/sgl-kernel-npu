import argparse
import inspect
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch_npu


def make_hccl_pg_options(buffer_size: Optional[int]):
    if buffer_size is None:
        return None
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": buffer_size}
    return options


def init_dist(
    local_rank: int, num_local_ranks: int, hccl_buffer_size: Optional[int] = None
):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    global_rank = node_rank * num_local_ranks + local_rank
    world_size = num_nodes * num_local_ranks

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=global_rank,
        pg_options=make_hccl_pg_options(hccl_buffer_size),
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    group = dist.new_group(
        list(range(world_size)), pg_options=make_hccl_pg_options(hccl_buffer_size)
    )

    return dist.get_rank(), dist.get_world_size(), group


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None, sync_fn=None):
    device = torch.device("npu")
    torch.npu.synchronize()

    # Flush L2 cache with 256 MB data
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(num_warmups):
        if sync_fn is not None:
            sync_fn()
        fn()
        if sync_fn is not None:
            sync_fn()

    # Flush L2 cache
    cache.zero_()
    torch.npu.synchronize()

    # Timing
    times = []
    for _ in range(num_tests):
        torch.npu.synchronize()

        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)

        if sync_fn is not None:
            sync_fn()

        start.record()
        fn()
        end.record()

        if sync_fn is not None:
            sync_fn()

        if post_fn is not None:
            post_fn()

        torch.npu.synchronize()
        elapsed_time = start.elapsed_time(end) / 1e3  # ms -> s
        times.append(elapsed_time)

    samples = times[1:] if len(times) >= 2 else times
    times = np.array(samples)
    return np.average(times), np.min(times), np.max(times)


def _fp4_e2m1_to_float32_unpack(fp4_bytes: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 E2M1 (fp4x2_e2m1fn_x2) packed bytes to float32 (all on CPU).
    Each byte contains 2 FP4 values: high nibble = element 0, low nibble = element 1.
    FP4 E2M1: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit (implicit leading 1).
    Special: exp=00,m=0 -> zero; exp=11 -> NaN
    """
    original_shape = fp4_bytes.shape
    flat = fp4_bytes.flatten().cpu().to(torch.int32)

    # Pre-compute all 16 FP4 E2M1 values as a lookup table
    fp4_table = torch.zeros(16, dtype=torch.float32)
    for i in range(16):
        sign = 1.0 if (i >> 3) == 0 else -1.0
        exp = (i & 0x7) >> 1
        mant = i & 1
        if exp == 0 and mant == 0:
            fp4_table[i] = 0.0
        else:
            fp4_table[i] = sign * (2.0 ** (exp - 1)) * (1.0 + mant * 0.5)

    high_nibbles = (flat >> 4) & 0xF
    low_nibbles = flat & 0xF
    vals_high = fp4_table[high_nibbles]
    vals_low = fp4_table[low_nibbles]

    result = torch.stack([vals_low, vals_high], dim=1).reshape(-1)
    new_shape = list(original_shape[:-1]) + [original_shape[-1] * 2]
    return result.reshape(new_shape)


_FP8E8M0_TO_FLOAT32_TABLE = None


def _fp8e8m0_to_float32_lookup(bits: torch.Tensor) -> torch.Tensor:
    global _FP8E8M0_TO_FLOAT32_TABLE
    if _FP8E8M0_TO_FLOAT32_TABLE is None:
        table = []
        for i in range(256):
            val = 2.0 ** (i - 127)
            table.append(val)
        _FP8E8M0_TO_FLOAT32_TABLE = torch.tensor(table, dtype=torch.float32)

    return _FP8E8M0_TO_FLOAT32_TABLE[bits.to(torch.long)]


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    # x_scales 现在是 FP8 E8M0 格式（uint8 或 int8 存储）
    # 需要先解码为 float32

    if x_scales.dtype != torch.float32:
        # 将存储的整数视为 FP8 E8M0 的位表示，转换为 float32
        x_scales_bits = x_scales.view(torch.uint8)
        x_scales_fp32 = _fp8e8m0_to_float32_lookup(x_scales_bits)

        if x_fp8.dtype == torch.float4_e2m1fn_x2:
            # FP4 dequant: each fp4x2 packs 2 elements, shape is (bs, h/2)
            # Scale is per 32 original elements: (bs * h/32)
            # Do entire dequant on CPU to avoid NPU FP4 dtype issues
            bs, h_half = x_fp8.shape
            h = h_half * 2
            x_fp4_uint8 = x_fp8.view(torch.uint8).cpu()
            x_fp32 = _fp4_e2m1_to_float32_unpack(x_fp4_uint8)  # CPU float32
            x_fp32 = x_fp32.view(bs, -1, 32)
            x_scales_bits_cpu = x_scales_bits.cpu()
            x_scales_fp32_cpu = _fp8e8m0_to_float32_lookup(x_scales_bits_cpu).view(
                bs, -1, 1
            )
            result_cpu = (x_fp32 * x_scales_fp32_cpu).view(bs, h).to(torch.bfloat16)
            return result_cpu.to(x_fp8.device)

        # x_fp8 形状: (bs, h)
        # x_scales 形状: (bs, h/32) 或 (bs * h/32,)
        bs, h = x_fp8.shape
        scale_per_32 = h // 32
        # 将 x_fp8 转为 float32 并 reshape 为 (bs, h/32, 32)
        x_fp32 = x_fp8.to(torch.float32).view(bs, -1, 32)
        typeinfo = torch.finfo(x_fp8.dtype)
        x_fp32 = torch.clamp(x_fp32, typeinfo.min, typeinfo.max)
        # 将 x_scales reshape 为 (bs, -1, 1) 用于广播
        x_scales_fp32 = x_scales_fp32.view(bs, -1, 1)
        # 逐元素乘法：每个 32 元素的组乘以对应的 scale
        result = (x_fp32 * x_scales_fp32).view(bs, h).to(torch.bfloat16)
        return result
    else:
        if x_scales.dtype == torch.int:
            x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
            x_scales = x_scales.view(dtype=torch.float)
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
        x_scales = x_scales.view(x_fp8.size(0), -1, 1)
        return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


DIFF_THRESHOLDS = {
    "bf16": 1e-5,
    "int8": 3e-3,
    "fp8": 2e-3,
    "fp4": 4e-2,
}


def get_diff_threshold(quant_type):
    if quant_type is None or quant_type in ("no", "bf16"):
        return DIFF_THRESHOLDS["bf16"]
    if quant_type == "int8":
        return DIFF_THRESHOLDS["int8"]
    if "fp4" in quant_type:
        return DIFF_THRESHOLDS["fp4"]
    if "fp8" in quant_type:
        return DIFF_THRESHOLDS["fp8"]
    return DIFF_THRESHOLDS["bf16"]


def generate_a3_weight_inputs(
    hidden: int,
    intermediate_hidden: int,
    num_local_experts: int,
    device: torch.device,
    weight_range: tuple[int, int],
    scale_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_low, weight_high = weight_range
    w13_weight = torch.randint(
        weight_low,
        weight_high,
        (num_local_experts, hidden, 2 * intermediate_hidden),
        dtype=torch.int8,
        device=device,
    )
    w2_weight = torch.randint(
        weight_low,
        weight_high,
        (num_local_experts, intermediate_hidden, hidden),
        dtype=torch.int8,
        device=device,
    )
    w13_weight_scale = torch.empty(
        (num_local_experts, 2 * intermediate_hidden),
        dtype=torch.float32,
        device=device,
    ).uniform_(*scale_range)
    w2_weight_scale = torch.empty(
        (num_local_experts, 1, hidden), dtype=torch.float32, device=device
    ).uniform_(*scale_range)
    return w13_weight, w2_weight, w13_weight_scale, w2_weight_scale


def validate_a3_fused_layer(
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    num_local_experts: int,
    expected_shape: tuple[int, int],
    expected_dtype: torch.dtype,
    layer_name: str,
) -> None:
    for weight in weights:
        if weight.dtype != expected_dtype or tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"Invalid {layer_name} weight shape/dtype: got {tuple(weight.shape)} "
                f"{weight.dtype}, expected {expected_shape} {expected_dtype}."
            )
    if len(weights) != num_local_experts:
        raise ValueError(
            f"{layer_name} weight list length mismatch: "
            f"got {len(weights)}, expected {num_local_experts}."
        )
    if len(scales) != num_local_experts:
        raise ValueError(
            f"{layer_name} scale list length mismatch: "
            f"got {len(scales)}, expected {num_local_experts}."
        )


MX_QUANT_MODE_NAMES = ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1")
# In torch_npu/op-plugin npu_moe_distribute_dispatch_v2, quant_mode=4 selects
# the generic MX branch; y_dtype determines the actual MX output format.
# Keep dtype names as strings so importing this module does not require MX dtypes.
_MX_QUANT_META = {
    "fp8_e4m3": {
        "dispatch_y_dtype": "torch.float8_e4m3fn",
        "quant_dst_type": "torch.float8_e4m3fn",
        "origin_dtype": "torch.float8_e4m3fn",
    },
    "fp8_e5m2": {
        "dispatch_y_dtype": "torch.float8_e5m2",
        "quant_dst_type": "torch.float8_e5m2",
        "origin_dtype": "torch.float8_e5m2",
    },
    "fp4_e2m1": {
        "dispatch_y_dtype": "torch_npu.float4_e2m1fn_x2",
        "quant_dst_type": "torch_npu.float4_e2m1fn_x2",
        "origin_dtype": "torch.float4_e2m1fn_x2",
    },
}


def get_mx_quant_config(
    quant_name: str, *, packed_fp4_activations: bool = False
) -> Dict[str, object]:
    config = {"dispatch_quant_mode": 4}
    dtype_meta = _MX_QUANT_META[quant_name]
    if packed_fp4_activations:
        dtype_meta = {**dtype_meta, "origin_dtype": "torch.uint8"}
    for key, dtype_path in dtype_meta.items():
        module_name, attribute_name = dtype_path.split(".", 1)
        module = torch if module_name == "torch" else torch_npu
        config[key] = getattr(module, attribute_name)
    return config


def fused_situ_reference(
    x: torch.Tensor, beta: Optional[float], linear_beta: Optional[float]
) -> torch.Tensor:
    # GMM1 already returns the input dtype, matching the fused epilogue's
    # half-precision rounding. Compute both branches and their product in FP32,
    # then round once before MX quantization.
    gate, up = x.float().chunk(2, dim=-1)
    beta = 4.0 if beta is None else beta
    # Follow the kernel's multiply-by-reciprocal and Exp/Add/Div order.
    # Multiplying by sigmoid is algebraically equivalent, but has different
    # FP32 rounding before BF16 and MX quantization.
    gate_out = beta * torch.tanh(gate * (1.0 / beta))
    gate_out = gate_out / (torch.exp(-gate) + 1.0)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up * (1.0 / linear_beta))
    return (gate_out * up).to(x.dtype)


def run_small_op_baseline(
    inputs: Dict[str, torch.Tensor],
    hcomm_name: str,
    rank: int,
    world_size: int,
    global_bs: int,
    args: argparse.Namespace,
    gmm_burn_in_repeats: int = 1,
    warmup_burn_in_buffers: Tuple[torch.Tensor, torch.Tensor] = None,
    situ_reference=fused_situ_reference,
    *,
    packed_fp4_weights: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp4_activations = packed_fp4_weights and args.quant == "fp4_e2m1"
    quant_cfg = get_mx_quant_config(args.quant, packed_fp4_activations=fp4_activations)
    output_dtype = inputs["x"].dtype
    activation_dtype = quant_cfg["origin_dtype"]
    scale_dtype = torch.uint8 if packed_fp4_weights else torch.float8_e8m0fnu
    weight_scale_arg = "scale"
    gmm_options = {}
    if packed_fp4_weights:
        gmm_options.update(
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        if fp4_activations:
            gmm_options.update(
                x_dtype=torch_npu.float4_e2m1fn_x2,
                scale_dtype=torch_npu.float8_e8m0fnu,
            )
        else:
            weight_scale_arg = "antiquant_scale"

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

    if gmm_burn_in_repeats > 1 and warmup_burn_in_buffers is not None:
        # This burn-in only runs on the first profiler warmup iteration.
        # It does not change the routed output; it only inserts extra device
        # work before the normal small-op chain of that warmup iteration.
        burn_in_lhs, burn_in_rhs = warmup_burn_in_buffers
        for _ in range(gmm_burn_in_repeats):
            _ = torch.matmul(burn_in_lhs, burn_in_rhs)

    def run_gmm(layer, activations, token_scales):
        return torch_npu.npu_grouped_matmul(
            x=[activations],
            weight=[inputs[f"gmm{layer}_weight_q"]],
            per_token_scale=[token_scales.view(scale_dtype)],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_token_nums,
            output_dtype=output_dtype,
            **{weight_scale_arg: [inputs[f"gmm{layer}_weight_scale"]]},
            **gmm_options,
        )[0]

    dynamic_scales = dynamic_scales.view(*dynamic_scales.shape[:-1], -1, 2)
    if fp4_activations:
        expand_x = expand_x.view(activation_dtype)
    y1_fp = run_gmm(1, expand_x, dynamic_scales)
    if args.activation == "situ":
        activation_out = situ_reference(
            y1_fp, beta=args.beta, linear_beta=args.linear_beta
        )
    else:
        activation_out = torch_npu.npu_swiglu(y1_fp)
    x2, x2_scale = torch_npu.npu_dynamic_mx_quant(
        activation_out, dst_type=quant_cfg["quant_dst_type"]
    )
    y2_fp = run_gmm(2, x2.view(activation_dtype), x2_scale)
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


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch_npu.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], schedule=schedule
        ) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="npu")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="npu")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="npu"))
                for _ in range(num_tests):
                    fn()
                torch.npu.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)

    kernel_names = (kernel_names,) if not is_tuple else kernel_names
    assert all(isinstance(name, str) for name in kernel_names)
    # Expand the kernels by periods

    # If the json file exists, `torch_npu.profiler.export_chrome_trace` will use the append write mode,
    # which will cause problems with the json format, so here we use a random file name instead of creating a temporary file
    temp_path = Path(tempfile.gettempdir()) / f"trace_{uuid.uuid4().hex}.json"
    prof.export_chrome_trace(temp_path)
    profile_data = json.loads(Path(temp_path).read_text())

    # Return average kernel durations
    kernel_durations = []
    for kernel_name in kernel_names:
        events = [event for event in profile_data if kernel_name in event["name"]]
        assert len(events) > 0, f"Kernel '{kernel_name}' not found in trace"
        events = sorted(events, key=lambda event: event["ts"])
        durations = [event["dur"] / 1e6 for event in events]
        if num_kernels_per_period > 1:
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations.append(
                [
                    sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                    for j in range(num_kernels_per_period)
                ]
            )
        else:
            num_kernel_patterns = len(durations)
            kernel_durations.append(sum(durations) / num_kernel_patterns)

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    os.unlink(temp_path)

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def _load_chrome_trace_events(trace_path: Union[str, Path]):
    profile_data = json.loads(Path(trace_path).read_text())
    if isinstance(profile_data, dict):
        profile_data = profile_data.get("traceEvents", [])
    return [
        event
        for event in profile_data
        if isinstance(event, dict)
        and "name" in event
        and "dur" in event
        and "ts" in event
        and isinstance(event["name"], str)
    ]


def _normalize_event_patterns(
    ordered_event_patterns: Union[str, tuple, list],
):
    if isinstance(ordered_event_patterns, str):
        ordered_event_patterns = [ordered_event_patterns]
    normalized = []
    for pattern in ordered_event_patterns:
        if isinstance(pattern, str):
            normalized.append((pattern,))
        else:
            normalized.append(tuple(pattern))
    return tuple(normalized)


def profile_npu_event_sequences(
    fn,
    ordered_event_patterns: Union[str, tuple, list],
    num_warmups: int = 0,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    allow_no_match: bool = False,
):
    if num_warmups < 0:
        raise ValueError("num_warmups must be non-negative")
    if num_tests <= 0:
        raise ValueError("num_tests must be positive")

    total_iters = num_warmups + num_tests
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch_npu.profiler.schedule(
            wait=0, warmup=0, active=total_iters, repeat=1
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], schedule=schedule
        ) as prof:
            for _ in range(total_iters):
                fn()
                prof.step()
            torch.npu.synchronize()

    temp_path = Path(tempfile.gettempdir()) / f"trace_{uuid.uuid4().hex}.json"
    prof.export_chrome_trace(temp_path)
    events = _load_chrome_trace_events(temp_path)

    if trace_path is not None:
        trace_path = Path(trace_path)
        if trace_path.exists():
            trace_path.unlink()
        prof.export_chrome_trace(trace_path)

    ordered_event_patterns = _normalize_event_patterns(ordered_event_patterns)
    discovered_names = sorted({event["name"] for event in events})
    events = sorted(events, key=lambda event: event["ts"])

    iteration_durations = []
    iteration_event_names = []
    iteration_step_durations = []
    current_duration = 0.0
    current_event_names = []
    current_step_durations = []
    current_pattern_idx = 0
    expected_patterns = len(ordered_event_patterns)

    for event in events:
        name = event["name"]
        if current_pattern_idx >= expected_patterns:
            current_pattern_idx = 0
            current_duration = 0.0
            current_event_names = []
            current_step_durations = []

        candidates = ordered_event_patterns[current_pattern_idx]
        if any(candidate in name for candidate in candidates):
            event_duration = event["dur"] / 1e6
            current_duration += event_duration
            current_event_names.append(name)
            current_step_durations.append(event_duration)
            current_pattern_idx += 1
            if current_pattern_idx == expected_patterns:
                iteration_durations.append(current_duration)
                iteration_event_names.append(tuple(current_event_names))
                iteration_step_durations.append(tuple(current_step_durations))
                current_duration = 0.0
                current_event_names = []
                current_step_durations = []
                current_pattern_idx = 0

    os.unlink(temp_path)

    if len(iteration_durations) == 0 and not allow_no_match:
        raise AssertionError(
            "No matched NPU event sequence found. "
            f"Patterns={ordered_event_patterns}. "
            f"Discovered events={discovered_names}"
        )

    if len(iteration_durations) < total_iters and not allow_no_match:
        raise AssertionError(
            "Matched NPU event iterations are fewer than expected. "
            f"Expected at least {total_iters}, got {len(iteration_durations)}. "
            f"Patterns={ordered_event_patterns}. "
            f"Discovered events={discovered_names}"
        )

    dropped_warmups = min(num_warmups, len(iteration_durations))
    if dropped_warmups:
        iteration_durations = iteration_durations[dropped_warmups:]
        iteration_event_names = iteration_event_names[dropped_warmups:]
        iteration_step_durations = iteration_step_durations[dropped_warmups:]

    durations = np.array(iteration_durations, dtype=np.float64)
    step_durations = np.array(iteration_step_durations, dtype=np.float64)
    debug_info = {
        "matched_total_iterations": len(iteration_durations) + dropped_warmups,
        "dropped_warmup_iterations": dropped_warmups,
        "counted_iterations": len(iteration_durations),
    }
    return (
        durations,
        discovered_names,
        iteration_event_names,
        step_durations,
        debug_info,
    )


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int8).sum().item()


def diagnose_matrix(
    mat,
    thres_col=3.0,
    thres_row=3.0,
    thres_point=5.0,
    suppress_points_in_strong_rowscols=True,
):
    """
    Detect abnormal columns, rows, and individual points in a 2D wait-time matrix.
    Arguments:
        mat (np.ndarray): 2D array where mat[i, j] is the waiting time of source i for destination j to
            receive(dispatch)/send(combine) the token
        thres_col/thres_row/thres_point(float): The ratio of the average waiting time for abnormal rank
            to the average waiting time for all ranks
        suppress_points_in_strong_rowscols (bool): If True, exclude points already in detected abnormal
            rows/columns.
    Returns:
        dict: {
            "abnormal_cols": List[List[int, float, float]],  # abnormal column indices
            "abnormal_rows": List[List[int, float, float]],  # abnormal row indices
            "abnormal_points": List[List[int, int, float, float]]  # abnormal points
        }
    """
    mat = mat.cpu().numpy()
    # 1. Check for abnormal columns
    col_means = mat.mean(axis=0)
    z_col = col_means / (col_means.mean() + 1e-8)
    abnormal_cols = [
        [j, col_means[j], z_col[j]] for j in np.where(z_col > thres_col)[0]
    ]

    # 2. Check for abnormal rows
    row_means = mat.mean(axis=1)
    z_row = row_means / (row_means.mean() + 1e-8)
    abnormal_rows = [
        [i, row_means[i], z_row[i]] for i in np.where(z_row > thres_row)[0]
    ]

    # 3. Check for abnormal single points
    z_all = mat / (mat.mean() + 1e-8)
    # Get all positions with z-score > threshold
    abnormal_points = [
        [i, j, mat[i, j], z_all[i, j]]
        for i in range(mat.shape[0])
        for j in range(mat.shape[1])
        if z_all[i, j] > thres_point
    ]
    # Optionally remove points that are in already detected abnormal rows
    # or columns
    if suppress_points_in_strong_rowscols:
        strong_rows = [row[0] for row in abnormal_rows]
        strong_cols = [col[0] for col in abnormal_cols]
        abnormal_points = [
            [i, j, v, z]
            for [i, j, v, z] in abnormal_points
            if i not in strong_rows and j not in strong_cols
        ]
    # 4. Return for automatic processing
    return {
        "abnormal_cols": abnormal_cols,
        "abnormal_rows": abnormal_rows,
        "abnormal_points": abnormal_points,
    }


def calculate_avg_stats(
    dispatch_t,
    num_dispatch_comm_bytes,
    combine_t,
    num_combine_comm_bytes,
    rank,
    num_ranks,
    root_rank: 0,
):
    # 1. 创建本地统计张量
    # 注意：确保 dtype 和 device 在所有进程中一致
    local_stats = torch.tensor(
        [
            dispatch_t * 1e6,  # us
            num_dispatch_comm_bytes,  # bytes
            combine_t * 1e6,  # us
            num_combine_comm_bytes,  # bytes
        ],
        dtype=torch.float64,
        device="npu",
    )

    # 2. 【修改点】为所有进程准备接收列表
    # all_gather 要求所有进程都参与，且缓冲区形状必须一致
    # 每个进程都创建一个大小为 num_ranks 的列表，每个元素形状与 local_stats 一致
    gather_list = [torch.zeros_like(local_stats) for _ in range(num_ranks)]

    # 3. 【修改点】使用 all_gather 替代 gather
    # 此时，每个进程的 gather_list 都将包含所有 rank 的 local_stats
    dist.all_gather(gather_list, local_stats)

    # 4. 仅在 root_rank (通常是 0) 上进行统计和打印
    # 其他进程虽然也执行了 all_gather，但不需要处理结果，节省计算资源
    if rank == root_rank:
        # 将列表堆叠成 [num_ranks, 4] 的张量
        stats_tensor = (
            torch.stack(gather_stats) if (gather_stats := gather_list) else None
        )
        # 注意：上面这行写法可能较新，为了兼容性，我们可以直接写：
        stats_tensor = torch.stack(gather_list)  # Shape [num_ranks, 4]

        dispatch_latency = stats_tensor[:, 0]  # us
        dispatch_bytes = stats_tensor[:, 1]  # bytes
        combine_latency = stats_tensor[:, 2]  # us
        combine_bytes = stats_tensor[:, 3]  # bytes

        # 计算平均值
        avg_dispatch_lat = torch.mean(dispatch_latency)
        avg_dispatch_bytes = torch.mean(dispatch_bytes)
        avg_combine_lat = torch.mean(combine_latency)
        avg_combine_bytes = torch.mean(combine_bytes)

        # 计算带宽 (GB/s)
        # 注意：如果 latency 为 0 会导致除以零错误，建议加一个极小值保护或断言
        # 这里假设 latency > 0
        avg_dispatch_bw = avg_dispatch_bytes / avg_dispatch_lat * 1e-3  # GB/s
        avg_combine_bw = avg_combine_bytes / avg_combine_lat * 1e-3  # GB/s

        print(
            f"\n\nAverage Dispatch bandwidth: {avg_dispatch_bw:.2f} GB/s, avg_t={avg_dispatch_lat:.2f} us \n"
            f"Average Combine bandwidth: {avg_combine_bw:.2f} GB/s, avg_t={avg_combine_lat:.2f} us\n\n",
            flush=True,
        )
