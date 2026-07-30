import argparse
import os
import random
from typing import List, Optional

import deep_ep
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from deep_ep.buffer import FuseMode
from utils import calc_diff, init_dist

torch_npu.npu.config.allow_internal_format = True
os.environ["MOE_EXPERT_TOKEN_NUMS_TYPE"] = "1"

ACL_FORMAT_FRACTAL_NZ = torch_npu.Format.FRACTAL_NZ
W4A8_DIFF_WARNING_THRESHOLD = 5e-3
EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0
EXPERT_TOKEN_NUMS_TYPE_COUNT = 1
COMM_QUANT_MODE_INT8 = 2


def info_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0] {message}", flush=True)


def warn_rank0(rank: int, message: str):
    if rank == 0:
        print(f"[rank0][WARNING] {message}", flush=True)


def summarize_value(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        parts = [summarize_value(item) for item in value[:4]]
        suffix = "" if len(value) <= 4 else ", ..."
        return f"{type(value).__name__}(len={len(value)}, items=[{', '.join(parts)}{suffix}])"
    return f"{type(value).__name__}({value})"


def set_env_if_provided(name: str, value):
    if value is not None:
        os.environ[name] = str(value)


def apply_runtime_env_from_args(args: argparse.Namespace):
    set_env_if_provided("MASTER_ADDR", args.master_addr)
    set_env_if_provided("MASTER_PORT", args.master_port)
    set_env_if_provided("WORLD_SIZE", args.num_servers)
    set_env_if_provided("RANK", args.server_index)
    set_env_if_provided("HCCL_BUFFSIZE", args.hccl_buffsize)


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_optional_float_list(raw: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for item in parse_csv_list(raw):
        value = float(item)
        values.append(None if value == 0 else value)
    return values


def parse_float_list(raw: str) -> List[float]:
    return [float(item) for item in parse_csv_list(raw)]


def swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


def situ_reference(
    x: torch.Tensor,
    *,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    if activation_clamp is not None and activation_clamp > 0:
        gate = torch.clamp(gate, -activation_clamp, activation_clamp)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None and linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def apply_activation(
    x: torch.Tensor,
    activation: str,
    *,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
) -> torch.Tensor:
    if activation == "swiglu":
        return torch_npu.npu_swiglu(x)
    if activation == "situ":
        return situ_reference(
            x,
            beta=beta,
            linear_beta=linear_beta,
            activation_clamp=activation_clamp,
        )
    raise ValueError(f"Unsupported activation: {activation}")


def make_topk_inputs(
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    return topk_ids.to(torch.int32), topk_weights


def validate_topk_inputs(topk_ids_int32: torch.Tensor, num_experts: int):
    if topk_ids_int32.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids_int32.dtype}")
    if torch.any(topk_ids_int32 < 0) or torch.any(topk_ids_int32 >= num_experts):
        raise ValueError("topk_ids contains out-of-range expert indices.")
    sorted_ids = torch.sort(topk_ids_int32, dim=-1).values
    if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("topk_ids must be unique within each token's top-k row.")


def pack_scale_to_uint64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64).view(torch.uint64)


def pack_scale_to_int64(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64)


def pack_int4(values: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Pack int8 values [-8,7] into int32 (8 values per int32)."""
    values = values.clamp(-8, 7)
    values_unsigned = torch.where(values < 0, values + 16, values).to(torch.int32)
    if pack_dim == 1:
        K, N_unpacked = values_unsigned.shape
        N_packed = N_unpacked // 8
        vals = values_unsigned.view(K, N_packed, 8)
    else:
        K_unpacked, N = values_unsigned.shape
        K_packed = K_unpacked // 8
        vals = values_unsigned.view(K_packed, 8, N).permute(0, 2, 1)
    packed = torch.zeros(
        vals.shape[0],
        vals.shape[1],
        dtype=torch.int32,
        device=values.device,
    )
    for i in range(8):
        packed |= (vals[..., i].to(torch.int32) << (i * 4)).to(torch.int32)
    return packed


def pack_int4_experts_to_int8(weights: torch.Tensor) -> torch.Tensor:
    if weights.dim() != 3:
        raise ValueError(f"Expected expert-stacked 3D weight, got {weights.shape}.")
    if weights.shape[-1] % 8 != 0:
        raise ValueError(
            f"logical int4 weight last dim must be divisible by 8, got {weights.shape}."
        )
    packed = [
        pack_int4(weight.contiguous(), 1).view(torch.int8)
        for weight in weights.unbind(dim=0)
    ]
    return torch.stack(packed, dim=0).contiguous()


def pack_int4_to_int8(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-1] % 2 != 0:
        raise ValueError(
            f"logical int4 weight last dim must be divisible by 2, got {weight.shape}."
        )
    packed = (
        (weight.to(torch.int16) + 8)
        .to(torch.uint8)
        .reshape(*weight.shape[:-1], weight.shape[-1] // 2, 2)
    )
    return ((packed[..., 1] << 4) | packed[..., 0]).view(torch.int8)


def normalize_expert_counts(
    counts, num_local_experts: int, device: torch.device
) -> torch.Tensor:
    if isinstance(counts, torch.Tensor):
        out = counts.to(device=device, dtype=torch.int64)
    else:
        out = torch.tensor(list(counts), device=device, dtype=torch.int64)
    if out.numel() != num_local_experts:
        raise ValueError(
            f"Expected {num_local_experts} local expert counts, got {out.numel()}."
        )
    return out


def pack_to_int32(weight: torch.Tensor, *, new_quant_version: bool) -> torch.Tensor:
    if not new_quant_version:
        raise ValueError("This test only supports new_quant_version=True for A3 W4A8.")
    if weight.shape[-1] % 4 != 0:
        raise ValueError(
            f"Packed int4 weight last dim must be divisible by 4, got {weight.shape}."
        )
    return weight.view(torch.int32)


def unpack_int4(packed: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Unpack int32 → int8. Returns int8 tensor."""
    K, N = packed.shape
    # 对每个移位单独操作，避免张量广播问题
    shifts = [0, 4, 8, 12, 16, 20, 24, 28]
    vals_list = []
    for shift in shifts:
        # 右移并取低4位
        val = (packed >> shift) & 0xF
        vals_list.append(val.unsqueeze(-1))
    vals = torch.cat(vals_list, dim=-1)  # [K, N, 8]

    vals = torch.where(vals >= 8, vals - 16, vals).to(torch.int8)

    if pack_dim == 1:
        out = vals.reshape(K, -1)
    else:
        out = vals.permute(0, 2, 1).contiguous().reshape(-1, N)
    return out


def pack_float_into_int64(dequant_scale_origin: torch.Tensor) -> torch.Tensor:
    """Pack float32 scale into int64 with marker bit."""
    scale_int = dequant_scale_origin.view(torch.int32)
    scale_int = scale_int & 0xFFFFE000  # clear low 13 mantissa bits
    # Convert to int64 without sign-extension: view as unsigned first
    out = torch.bitwise_and(
        scale_int.to(torch.int64),
        torch.tensor(
            0xFFFFFFFF,
            dtype=torch.int64,
            device=scale_int.device,
        ),
    )
    return out | (1 << 46)


def extract_float_from_int64(packed_int64: torch.Tensor) -> torch.Tensor:
    """Extract low 32 bits of int64 as float32."""
    return packed_int64.to(torch.int32).view(torch.float32)


def compute_bias(
    weight_packed: torch.Tensor, scale_packed: torch.Tensor, pack_dim: int = 1
) -> torch.Tensor:
    """Compute bias = sum(unpacked_weight * scale, dim=0) * 8."""
    if weight_packed.dtype != torch.int32:
        raise TypeError(
            f"compute_bias expects int32 packed weight, got {weight_packed.dtype}"
        )
    w_int4 = unpack_int4(weight_packed, pack_dim).float()
    scale_fp32 = extract_float_from_int64(scale_packed)
    return (w_int4 * scale_fp32).sum(dim=0) * 8.0


def convert_cumsum_group_list_to_count(group_list: torch.Tensor) -> torch.Tensor:
    if group_list.dim() != 1:
        raise ValueError(f"group_list must be 1D, got {tuple(group_list.shape)}.")
    return torch.cat([group_list[:1], torch.diff(group_list, dim=0)])


def build_scene_weight_source(
    hidden: int,
    intermediate_hidden: int,
    num_local_experts: int,
    device: torch.device,
) -> dict:
    new_quant_version = True

    # Generate logical int4 weights in int8 containers first. These tensors are
    # not packed yet; the last dim still represents the full output-channel dim.
    w13_weight_raw_int4 = torch.randint(
        -8,
        8,
        (num_local_experts, hidden, 2 * intermediate_hidden),
        dtype=torch.int8,
        device=device,
    )
    w2_weight_raw_int4 = torch.randint(
        -8,
        8,
        (num_local_experts, intermediate_hidden, hidden),
        dtype=torch.int8,
        device=device,
    )
    w13_weight_scale = torch.empty(
        (num_local_experts, 2 * intermediate_hidden),
        dtype=torch.float32,
        device=device,
    ).uniform_(0.01, 0.1)
    w2_weight_scale = torch.empty(
        (num_local_experts, 1, hidden), dtype=torch.float32, device=device
    ).uniform_(0.01, 0.1)
    w13_weight_packed_int8 = pack_int4_experts_to_int8(w13_weight_raw_int4)
    w2_weight_packed_int8 = pack_int4_experts_to_int8(w2_weight_raw_int4)
    expected_w13_packed_int8_shape = (
        num_local_experts,
        hidden,
        intermediate_hidden,
    )
    expected_w2_packed_int8_shape = (
        num_local_experts,
        intermediate_hidden,
        hidden // 2,
    )
    if tuple(w13_weight_packed_int8.shape) != expected_w13_packed_int8_shape:
        raise ValueError(
            f"Invalid packed int8 w13 shape: got {tuple(w13_weight_packed_int8.shape)}, "
            f"expected {expected_w13_packed_int8_shape}."
        )
    if tuple(w2_weight_packed_int8.shape) != expected_w2_packed_int8_shape:
        raise ValueError(
            f"Invalid packed int8 w2 shape: got {tuple(w2_weight_packed_int8.shape)}, "
            f"expected {expected_w2_packed_int8_shape}."
        )
    w13_scale_int64 = pack_float_into_int64(w13_weight_scale.contiguous())
    w2_scale_int64 = pack_float_into_int64(w2_weight_scale.contiguous())
    expected_w13_scale_shape = (num_local_experts, 2 * intermediate_hidden)
    expected_w2_scale_shape = (num_local_experts, 1, hidden)
    if tuple(w13_scale_int64.shape) != expected_w13_scale_shape:
        raise ValueError(
            f"Invalid packed int64 w13 scale shape: got {tuple(w13_scale_int64.shape)}, "
            f"expected {expected_w13_scale_shape}."
        )
    if tuple(w2_scale_int64.shape) != expected_w2_scale_shape:
        raise ValueError(
            f"Invalid packed int64 w2 scale shape: got {tuple(w2_scale_int64.shape)}, "
            f"expected {expected_w2_scale_shape}."
        )
    w13_bias = torch.stack(
        [
            compute_bias(
                pack_to_int32(w.contiguous(), new_quant_version=new_quant_version), s, 1
            )
            for w, s in zip(
                w13_weight_packed_int8.unbind(dim=0), w13_scale_int64.unbind(dim=0)
            )
        ],
        dim=0,
    ).to(torch.float32)
    w2_bias = torch.stack(
        [
            compute_bias(
                pack_to_int32(w.contiguous(), new_quant_version=new_quant_version), s, 1
            )
            for w, s in zip(
                w2_weight_packed_int8.unbind(dim=0), w2_scale_int64.unbind(dim=0)
            )
        ],
        dim=0,
    ).to(torch.float32)
    expected_w13_bias_shape = (num_local_experts, 2 * intermediate_hidden)
    expected_w2_bias_shape = (num_local_experts, hidden)
    if tuple(w13_bias.shape) != expected_w13_bias_shape:
        raise ValueError(
            f"Invalid computed w13 bias shape: got {tuple(w13_bias.shape)}, "
            f"expected {expected_w13_bias_shape}."
        )
    if tuple(w2_bias.shape) != expected_w2_bias_shape:
        raise ValueError(
            f"Invalid computed w2 bias shape: got {tuple(w2_bias.shape)}, "
            f"expected {expected_w2_bias_shape}."
        )
    return {
        "hidden": hidden,
        "intermediate_hidden": intermediate_hidden,
        "num_local_experts": num_local_experts,
        "new_quant_version": new_quant_version,
        "w13_weight_raw_int4": w13_weight_raw_int4,
        "w2_weight_raw_int4": w2_weight_raw_int4,
        "w13_weight_packed_int8": w13_weight_packed_int8,
        "w2_weight_packed_int8": w2_weight_packed_int8,
        "w13_scale_int64": w13_scale_int64,
        "w2_scale_int64": w2_scale_int64,
        "w13_bias": w13_bias,
        "w2_bias": w2_bias,
    }


def build_baseline_scene_weights(source: dict, device: torch.device) -> dict:
    new_quant_version = source["new_quant_version"]

    w13_weight_baseline_nz = torch_npu.npu_format_cast(
        source["w13_weight_packed_int8"].to(device), ACL_FORMAT_FRACTAL_NZ
    )
    w2_weight_baseline_nz = torch_npu.npu_format_cast(
        source["w2_weight_packed_int8"].to(device), ACL_FORMAT_FRACTAL_NZ
    )
    w13_weight_packed_stacked = pack_to_int32(
        w13_weight_baseline_nz, new_quant_version=new_quant_version
    )
    w2_weight_packed_stacked = pack_to_int32(
        w2_weight_baseline_nz, new_quant_version=new_quant_version
    )

    return {
        "baseline_l1_weight_stacked": [w13_weight_packed_stacked],
        "baseline_l2_weight_stacked": [w2_weight_packed_stacked],
        "baseline_l1_scale_stacked": [
            source["w13_scale_int64"].to(device).unsqueeze(1).contiguous()
        ],
        "baseline_l2_scale_stacked": [source["w2_scale_int64"].to(device).contiguous()],
        "baseline_l1_bias_stacked": [source["w13_bias"].to(device).contiguous()],
        "baseline_l2_bias_stacked": [source["w2_bias"].to(device).contiguous()],
    }


def build_fused_scene_weights(source: dict, device: torch.device) -> dict:
    hidden = source["hidden"]
    intermediate_hidden = source["intermediate_hidden"]
    num_local_experts = source["num_local_experts"]
    new_quant_version = source["new_quant_version"]

    fused_l1_weights = [
        pack_to_int32(
            torch_npu.npu_format_cast(
                weight.contiguous().to(device), ACL_FORMAT_FRACTAL_NZ
            ),
            new_quant_version=new_quant_version,
        )
        for weight in source["w13_weight_packed_int8"].unbind(dim=0)
    ]
    fused_l2_weights = [
        pack_to_int32(
            torch_npu.npu_format_cast(
                weight.contiguous().to(device), ACL_FORMAT_FRACTAL_NZ
            ),
            new_quant_version=new_quant_version,
        )
        for weight in source["w2_weight_packed_int8"].unbind(dim=0)
    ]
    fused_l1_scales = [
        scale.to(device).reshape(-1).view(torch.uint64)
        for scale in source["w13_scale_int64"].unbind(dim=0)
    ]
    fused_l2_scales = [
        scale.to(device).reshape(-1).view(torch.uint64)
        for scale in source["w2_scale_int64"].unbind(dim=0)
    ]
    fused_l1_bias = [
        bias.to(device).reshape(-1).to(torch.float32)
        for bias in source["w13_bias"].unbind(dim=0)
    ]
    fused_l2_bias = [
        bias.to(device).reshape(-1).to(torch.float32)
        for bias in source["w2_bias"].unbind(dim=0)
    ]

    expected_l1_shape = (hidden, (2 * intermediate_hidden) // 8)
    expected_l2_shape = (intermediate_hidden, hidden // 8)
    for weight in fused_l1_weights:
        if weight.dtype != torch.int32 or tuple(weight.shape) != expected_l1_shape:
            raise ValueError(
                f"Invalid fused l1 weight shape/dtype: got {tuple(weight.shape)} {weight.dtype}, "
                f"expected {expected_l1_shape} torch.int32."
            )
    for weight in fused_l2_weights:
        if weight.dtype != torch.int32 or tuple(weight.shape) != expected_l2_shape:
            raise ValueError(
                f"Invalid fused l2 weight shape/dtype: got {tuple(weight.shape)} {weight.dtype}, "
                f"expected {expected_l2_shape} torch.int32."
            )
    if (
        len(fused_l1_weights) != num_local_experts
        or len(fused_l2_weights) != num_local_experts
    ):
        raise ValueError("Fused weight list length mismatch.")
    if (
        len(fused_l1_scales) != num_local_experts
        or len(fused_l2_scales) != num_local_experts
    ):
        raise ValueError("Fused scale list length mismatch.")
    if (
        len(fused_l1_bias) != num_local_experts
        or len(fused_l2_bias) != num_local_experts
    ):
        raise ValueError("Fused bias list length mismatch.")

    return {
        "fused_l1_weights": fused_l1_weights,
        "fused_l2_weights": fused_l2_weights,
        "fused_l1_scales": fused_l1_scales,
        "fused_l2_scales": fused_l2_scales,
        "fused_l1_bias": fused_l1_bias,
        "fused_l2_bias": fused_l2_bias,
    }


def prepare_scene_weights(
    hidden: int,
    intermediate_hidden: int,
    num_local_experts: int,
    device: torch.device,
) -> dict:
    # The generation of weights and scales for w4a8 must be performed on the CPU;
    # using the device for these calculations leads to precision mismatches between small operators and large fused operators.
    source = build_scene_weight_source(
        hidden,
        intermediate_hidden,
        num_local_experts,
        torch.device("cpu"),
    )
    baseline_weights = build_baseline_scene_weights(source, device)
    print(
        "[build_fused_scene_weights][source] "
        + " | ".join(
            f"{key}={summarize_value(value)}" for key, value in source.items()
        ),
        flush=True,
    )
    fused_weights = build_fused_scene_weights(source, device)
    return {
        "source": source,
        **baseline_weights,
        **fused_weights,
        "dispatch_quant_mode": 2,
        "dispatch_quant_out_dtype": torch.int8,
    }


def run_grouped_matmul_w4a8(
    x_int8: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    weight: List[torch.Tensor],
    scale: List[torch.Tensor],
    bias,
    rank: int,
    stage_name: str,
    barrier_group: dist.ProcessGroup,
) -> torch.Tensor:
    bias_list = bias if isinstance(bias, list) else [bias]
    out = torch_npu.npu_grouped_matmul(
        x=[x_int8],
        weight=weight,
        scale=[scale[0].to(scale[0].dtype)],
        bias=bias_list,
        per_token_scale=[per_token_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    return out


def run_mc2_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    num_experts: int,
    global_bs: int,
    x_active_mask: Optional[torch.Tensor],
    group_ep: str,
    barrier_group: dist.ProcessGroup,
    ep_world_size: int,
    ep_rank_id: int,
    enable_dispatch_v2: bool,
):
    kwargs = {
        "x": x,
        "expert_ids": topk_idx,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": num_experts,
        "global_bs": global_bs,
        "x_active_mask": x_active_mask,
        "quant_mode": COMM_QUANT_MODE_INT8,
        "scales": None,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "expert_token_nums_type": EXPERT_TOKEN_NUMS_TYPE_COUNT,
    }
    output = (
        torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        if enable_dispatch_v2
        else torch_npu.npu_moe_distribute_dispatch(**kwargs)
    )
    return output[0:7]


def run_mc2_combine(
    mlp_out: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    global_bs: int,
    x_active_mask: Optional[torch.Tensor],
    ep_recv_counts: torch.Tensor,
    tp_recv_counts: torch.Tensor,
    assist_info_for_combine,
    expand_scales: torch.Tensor,
    group_ep: str,
    barrier_group: dist.ProcessGroup,
    ep_world_size: int,
    ep_rank_id: int,
    enable_dispatch_v2: bool,
) -> torch.Tensor:
    kwargs = {
        "expand_x": mlp_out,
        "expert_ids": topk_idx,
        "expert_scales": topk_weights.to(torch.float32),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": num_experts,
        "global_bs": global_bs,
        "ep_send_counts": ep_recv_counts,
        "x_active_mask": x_active_mask,
        "group_ep": group_ep,
        "ep_world_size": ep_world_size,
        "ep_rank_id": ep_rank_id,
        "expand_scales": expand_scales,
        "comm_quant_mode": 0,
    }
    if enable_dispatch_v2:
        kwargs["assist_info_for_combine"] = assist_info_for_combine
        out = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
    else:
        kwargs["expand_idx"] = assist_info_for_combine
        out = torch_npu.npu_moe_distribute_combine(**kwargs)
    return out


def run_baseline_reference(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    weights: dict,
    activation: str,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
    group_ep: str,
    barrier_group: dist.ProcessGroup,
    ep_world_size: int,
    ep_rank_id: int,
    enable_dispatch_v2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_num_tokens = x.size(0)
    max_num_tokens = 32
    global_bs = max_num_tokens * ep_world_size

    x_active_mask = torch.zeros(
        max_num_tokens,
        dtype=torch.bool,
        device=x.device,
    )
    x_active_mask[:local_num_tokens] = True

    if local_num_tokens == max_num_tokens:
        x_padded = x
        topk_idx_padded = topk_idx
        topk_weights_padded = topk_weights
    else:
        padding_size = max_num_tokens - local_num_tokens
        x_padded = torch.cat(
            (x, x.new_zeros((padding_size, x.size(1)))),
            dim=0,
        )
        topk_idx_padded = torch.cat(
            (
                topk_idx,
                topk_idx.new_zeros((padding_size, topk_idx.size(1))),
            ),
            dim=0,
        )
        topk_weights_padded = torch.cat(
            (
                topk_weights,
                topk_weights.new_zeros((padding_size, topk_weights.size(1))),
            ),
            dim=0,
        )

    (
        recv_x_int8,
        recv_x_scale,
        assist_info_for_combine,
        group_list,
        ep_recv_counts,
        tp_recv_counts,
        expand_scales,
    ) = run_mc2_dispatch(
        x_padded,
        topk_idx_padded,
        num_experts=num_experts,
        global_bs=global_bs,
        x_active_mask=x_active_mask,
        group_ep=group_ep,
        barrier_group=barrier_group,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        enable_dispatch_v2=enable_dispatch_v2,
    )
    group_list_type = EXPERT_TOKEN_NUMS_TYPE_COUNT
    if group_list.numel() != num_experts // dist.get_world_size():
        group_list = convert_cumsum_group_list_to_count(group_list)

    gate_up = run_grouped_matmul_w4a8(
        recv_x_int8,
        recv_x_scale,
        group_list,
        group_list_type,
        weights["baseline_l1_weight_stacked"],
        weights["baseline_l1_scale_stacked"],
        weights["baseline_l1_bias_stacked"],
        ep_rank_id,
        "gmm1",
        barrier_group,
    )
    act_out = apply_activation(
        gate_up,
        activation,
        beta=beta,
        linear_beta=linear_beta,
        activation_clamp=activation_clamp,
    )
    act_int8, act_scale = torch_npu.npu_dynamic_quant(act_out)
    down = run_grouped_matmul_w4a8(
        act_int8,
        act_scale,
        group_list,
        group_list_type,
        weights["baseline_l2_weight_stacked"],
        weights["baseline_l2_scale_stacked"],
        weights["baseline_l2_bias_stacked"],
        ep_rank_id,
        "gmm2",
        barrier_group,
    )
    combined_x = run_mc2_combine(
        down,
        topk_idx_padded,
        topk_weights_padded,
        num_experts=num_experts,
        global_bs=global_bs,
        x_active_mask=x_active_mask,
        ep_recv_counts=ep_recv_counts,
        tp_recv_counts=tp_recv_counts,
        assist_info_for_combine=assist_info_for_combine,
        expand_scales=expand_scales,
        group_ep=group_ep,
        barrier_group=barrier_group,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        enable_dispatch_v2=enable_dispatch_v2,
    )
    return combined_x[:local_num_tokens], normalize_expert_counts(
        group_list,
        num_experts // dist.get_world_size(),
        x.device,
    )


def run_fused_reference(
    buffer: deep_ep.Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    weights: dict,
    activation: str,
    beta: float,
    linear_beta: Optional[float],
    activation_clamp: Optional[float],
    barrier_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = buffer.fused_deep_moe(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        gmm1_permuted_weight=weights["fused_l1_weights"],
        gmm1_permuted_weight_scale=weights["fused_l1_scales"],
        gmm2_weight=weights["fused_l2_weights"],
        gmm2_weight_scale=weights["fused_l2_scales"],
        num_max_dispatch_tokens_per_rank=32,
        num_experts=num_experts,
        backend="mega_moe",
        fuse_mode=FuseMode.FUSED_DEEP_MOE,
        activation=activation,
        activation_clamp=activation_clamp,
        beta=beta,
        linear_beta=linear_beta,
        l1_bias=weights["fused_l1_bias"],
        l2_bias=weights["fused_l2_bias"],
        dispatch_quant_mode=weights["dispatch_quant_mode"],
        dispatch_quant_out_dtype=weights["dispatch_quant_out_dtype"],
    )
    return out


def build_case_matrix(args: argparse.Namespace) -> list[dict]:
    activations = parse_csv_list(args.activation)
    betas = parse_float_list(args.beta)
    linear_betas = parse_optional_float_list(args.linear_beta)
    activation_clamps = parse_optional_float_list(args.activation_clamp)

    cases = []
    for activation in activations:
        if activation == "swiglu":
            cases.append(
                {
                    "activation": "swiglu",
                    "beta": 1.0,
                    "linear_beta": None,
                    "activation_clamp": None,
                }
            )
            continue
        if activation != "situ":
            raise ValueError(f"Unsupported activation case: {activation}")
        for beta in betas:
            for linear_beta in linear_betas:
                for activation_clamp in activation_clamps:
                    cases.append(
                        {
                            "activation": "situ",
                            "beta": beta,
                            "linear_beta": linear_beta,
                            "activation_clamp": activation_clamp,
                        }
                    )
    return cases


def launch_case(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    fused_buffer: deep_ep.Buffer,
    weights: dict,
    case: dict,
    iteration: int,
    case_index: int,
    baseline_group_ep: str,
    baseline_group: dist.ProcessGroup,
    mega_group: dist.ProcessGroup,
    enable_dispatch_v2: bool,
):
    device = torch.device(f"npu:{torch.npu.current_device()}")
    num_local_experts = args.num_experts // num_ranks

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    x = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = make_topk_inputs(
        args.num_tokens, args.num_experts, args.num_topk, device
    )
    baseline_out, baseline_counts = run_baseline_reference(
        x,
        topk_idx,
        topk_weights,
        args.num_experts,
        weights,
        case["activation"],
        case["beta"],
        case["linear_beta"],
        case["activation_clamp"],
        baseline_group_ep,
        baseline_group,
        num_ranks,
        rank,
        enable_dispatch_v2,
    )
    fused_out, fused_counts = run_fused_reference(
        fused_buffer,
        x,
        topk_idx,
        topk_weights,
        args.num_tokens,
        args.num_experts,
        weights,
        case["activation"],
        case["beta"],
        case["linear_beta"],
        case["activation_clamp"],
        mega_group,
    )
    info_rank0(
        rank,
        f"iter={iteration} case={case_index} activation={case['activation']} "
        f"baseline_out_shape={tuple(baseline_out.shape)} "
        f"fused_out_shape={tuple(fused_out.shape)}",
    )

    return {
        "case": case,
        "iteration": iteration,
        "case_index": case_index,
        "device": device,
        "num_local_experts": num_local_experts,
        "x": x,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "weights": weights,
        "baseline_out": baseline_out,
        "baseline_counts": baseline_counts,
        "fused_out": fused_out,
        "fused_counts": fused_counts,
    }


def finalize_case(
    args: argparse.Namespace,
    rank: int,
    launched_case: dict,
    synchronize_ranks: bool = True,
):
    case = launched_case["case"]
    iteration = launched_case["iteration"]
    case_index = launched_case["case_index"]
    device = launched_case["device"]
    num_local_experts = launched_case["num_local_experts"]
    x = launched_case["x"]
    topk_idx = launched_case["topk_idx"]
    topk_weights = launched_case["topk_weights"]
    weights = launched_case["weights"]
    baseline_out = launched_case["baseline_out"]
    baseline_counts = launched_case["baseline_counts"]
    fused_out = launched_case["fused_out"]
    fused_counts = launched_case["fused_counts"]

    local_diff = calc_diff(
        baseline_out.float(),
        fused_out.float(),
    )
    diff_tensor = torch.tensor(local_diff, dtype=torch.float32, device=device)
    dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)

    counts_match = torch.equal(
        normalize_expert_counts(baseline_counts, num_local_experts, device),
        normalize_expert_counts(fused_counts, num_local_experts, device),
    )
    counts_mismatch = torch.tensor(
        0 if counts_match else 1, dtype=torch.int32, device=device
    )
    dist.all_reduce(counts_mismatch, op=dist.ReduceOp.MAX)

    case_desc = (
        f"iter={iteration} activation={case['activation']} beta={case['beta']} "
        f"linear_beta={case['linear_beta']} activation_clamp={case['activation_clamp']}"
    )
    if counts_mismatch.item() != 0:
        warn_rank0(rank, f"{case_desc} expert_token_nums mismatch across ranks.")
    max_diff = diff_tensor.item()
    if max_diff >= W4A8_DIFF_WARNING_THRESHOLD:
        warn_rank0(
            rank,
            f"{case_desc} calc_diff={max_diff} exceeds threshold={W4A8_DIFF_WARNING_THRESHOLD}",
        )
    else:
        info_rank0(
            rank,
            f"{case_desc} calc_diff={max_diff} token_counts_match={counts_mismatch.item() == 0}",
        )
    if synchronize_ranks:
        dist.barrier()


def test_main(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    fused_buffer: deep_ep.Buffer,
    baseline_group_ep: str,
    mega_group_ep: str,
    baseline_group: dist.ProcessGroup,
    mega_group: dist.ProcessGroup,
    enable_dispatch_v2: bool,
):
    if args.num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by world_size.")
    if args.enable_performance and args.performance_iters <= 0:
        raise ValueError("--performance-iters must be greater than zero.")

    iterations = args.performance_iters if args.enable_performance else 1
    cases = build_case_matrix(args)
    device = torch.device(f"npu:{torch.npu.current_device()}")
    num_local_experts = args.num_experts // num_ranks
    weights = prepare_scene_weights(
        args.hidden,
        args.moe_intermediate_size,
        num_local_experts,
        device,
    )
    info_rank0(
        rank,
        f"baseline_group_ep={baseline_group_ep} mega_group_ep={mega_group_ep}",
    )
    if args.enable_performance:
        launched_cases = []
        for iteration in range(iterations):
            for case_index, case in enumerate(cases):
                launched_cases.append(
                    launch_case(
                        args,
                        rank,
                        num_ranks,
                        fused_buffer,
                        weights,
                        case,
                        iteration,
                        case_index,
                        baseline_group_ep,
                        baseline_group,
                        mega_group,
                        enable_dispatch_v2,
                    )
                )

        torch.npu.synchronize()
        dist.barrier()
        for launched_case in launched_cases:
            finalize_case(args, rank, launched_case, synchronize_ranks=False)
    else:
        for iteration in range(iterations):
            for case_index, case in enumerate(cases):
                finalize_case(
                    args,
                    rank,
                    launch_case(
                        args,
                        rank,
                        num_ranks,
                        fused_buffer,
                        weights,
                        case,
                        iteration,
                        case_index,
                        baseline_group_ep,
                        baseline_group,
                        mega_group,
                        enable_dispatch_v2,
                    ),
                )
    info_rank0(rank, "A3 W4A8 fused_deep_moe accuracy test completed")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, _ = init_dist(local_rank, num_local_ranks)
    ranks = list(range(num_ranks))
    baseline_group = dist.new_group(ranks=ranks, backend="hccl")
    mega_group = dist.new_group(ranks=ranks, backend="hccl")
    baseline_backend = baseline_group._get_backend(torch.device("npu"))
    mega_backend = mega_group._get_backend(torch.device("npu"))
    baseline_group_ep = baseline_backend.get_hccl_comm_name(rank)
    mega_group_ep = mega_backend.get_hccl_comm_name(rank)
    enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")

    # if rank == 0:
    #     args.num_tokens = 32
    # else:
    #     args.num_tokens = 1

    fused_buffer = deep_ep.Buffer(
        mega_group,
        int(2e9),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )

    test_main(
        args,
        rank,
        num_ranks,
        fused_buffer,
        baseline_group_ep,
        mega_group_ep,
        baseline_group,
        mega_group,
        enable_dispatch_v2,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A3 W4A8 fused_deep_moe vs MC2 small-op accuracy test."
    )
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument(
        "--num-ranks-per-server",
        type=int,
        default=None,
        help=(
            "Number of local ranks spawned on the current server. Defaults to "
            "--num-processes and must match it for this launcher."
        ),
    )
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="Set MASTER_ADDR before spawning local worker processes.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Set MASTER_PORT before spawning local worker processes.",
    )
    parser.add_argument(
        "--hccl-buffsize",
        type=int,
        default=None,
        help="Set HCCL_BUFFSIZE before spawning.",
    )
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--moe-intermediate-size", type=int, default=3072)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--activation", type=str, default="swiglu,situ")
    parser.add_argument("--beta", type=str, default="4.0")
    parser.add_argument("--linear-beta", type=str, default="25.0")
    parser.add_argument("--activation-clamp", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-performance", action="store_true")
    parser.add_argument("--performance-iters", type=int, default=30)
    args = parser.parse_args()
    if args.num_ranks_per_server is None:
        args.num_ranks_per_server = args.num_processes
    if args.num_ranks_per_server != args.num_processes:
        raise ValueError(
            "--num-ranks-per-server must equal --num-processes because this "
            "script starts one local worker per process."
        )

    apply_runtime_env_from_args(args)
    torch.multiprocessing.spawn(
        test_loop,
        args=(args.num_ranks_per_server, args),
        nprocs=args.num_processes,
    )
