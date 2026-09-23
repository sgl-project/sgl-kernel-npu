"""Replay real mode-2 inputs without replacing the model's fused outputs.

Used only by diagnose_sglang_fuseep.py. Separate HCCL groups/windows keep the
reference from reusing the fused operator's communication state.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer

_buffer = None
_seen = set()
_scales = {}


def unpack_scale(value, experts):
    assert value.dtype == torch.int64, value.dtype
    return (
        value.detach()
        .cpu()
        .to(torch.int32)
        .view(torch.float32)
        .reshape(experts, -1)
        .to(value.device)
    )


def reference(buffer, x, ids, probs, layer, scales, variant):
    (qx, sx), counts, handle, event, hook = (
        buffer.low_latency_strategy.low_latency_dispatch(
            x,
            ids.to(torch.int64),
            128,
            layer.num_experts,
            use_fp8=False,
            quant_mode="int8",
            async_finish=False,
            return_recv_hook=False,
        )
    )
    assert qx.dtype == torch.int8 and sx.dtype == torch.float32
    group_list = counts.to(torch.int64)
    kwargs = dict(split_item=2, group_list_type=1, group_type=0, group_list=group_list)
    if variant == "int32_dequant_swiglu":
        h = torch_npu.npu_grouped_matmul(
            x=[qx],
            weight=[layer.w13_weight],
            output_dtype=torch.int32,
            **kwargs,
        )[0]
        qh, sh = torch_npu.npu_dequant_swiglu_quant(
            x=h,
            weight_scale=scales[0],
            activation_scale=sx,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
        )
    else:
        h = torch_npu.npu_grouped_matmul(
            x=[qx],
            weight=[layer.w13_weight],
            scale=[scales[0].bfloat16()],
            per_token_scale=[sx],
            output_dtype=torch.bfloat16,
            **kwargs,
        )[0]
        qh, sh = torch_npu.npu_dequant_swiglu_quant(h, activate_left=True, quant_mode=1)
    out = torch_npu.npu_grouped_matmul(
        x=[qh],
        weight=[layer.w2_weight],
        scale=[scales[1].bfloat16()],
        per_token_scale=[sh],
        output_dtype=torch.bfloat16,
        **kwargs,
    )[0]
    out, event, hook = buffer.low_latency_combine(
        out,
        ids.to(torch.int64),
        probs,
        handle,
        async_finish=False,
        return_recv_hook=False,
    )
    torch.npu.synchronize()
    return out, counts


def metrics(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if a.numel() == 0:
        return dict(empty=True, reference_finite=True, candidate_finite=True)
    delta = (a - b).abs()
    denom = (a.square().sum() + b.square().sum()).clamp_min(1e-30)
    return dict(
        mean_abs=delta.mean().item(),
        max_abs=delta.max().item(),
        relative_squared_error=(delta.square().sum() / denom).item(),
        reference_abs_mean=a.abs().mean().item(),
        candidate_abs_mean=b.abs().mean().item(),
        reference_finite=bool(a.isfinite().all()),
        candidate_finite=bool(b.isfinite().all()),
    )


def audit(layer, fused_buffer, x, topk, out, counts):
    global _buffer
    layer_id = layer.layer_id
    max_tokens = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int64)
    dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=fused_buffer.group)
    token_count = int(max_tokens.item())
    key = (layer_id, token_count)
    if key in _seen or token_count < int(
        os.environ.get("SGLANG_FUSEEP_AUDIT_MIN_TOKENS", "4")
    ):
        return
    assert max_tokens.item() <= 128
    torch.npu.synchronize()
    if _buffer is None:
        ranks = dist.get_process_group_ranks(fused_buffer.group)
        group = dist.new_group(ranks=ranks, backend="hccl")
        _buffer = Buffer(
            group,
            low_latency_mode=True,
            num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(
                128, x.shape[1], len(ranks), layer.num_experts
            ),
            num_qps_per_rank=layer.num_experts // len(ranks),
        )
    if layer_id not in _scales:
        experts = layer.w13_weight.shape[0]
        _scales[layer_id] = (
            unpack_scale(layer.w13_weight_scale, experts),
            unpack_scale(layer.w2_weight_scale, experts),
        )
    scales = _scales[layer_id]
    row = dict(
        layer=layer_id,
        rank=dist.get_rank(fused_buffer.group),
        input_shape=list(x.shape),
        max_tokens_across_ranks=token_count,
        input_abs_max=x.detach().float().abs().max().item() if x.numel() else 0.0,
        weight13_shape=list(layer.w13_weight.shape),
        weight2_shape=list(layer.w2_weight.shape),
    )
    references = []
    for variant in ["int32_dequant_swiglu", "bf16_gmm_swiglu"]:
        ref, ref_counts = reference(
            _buffer, x, topk.topk_ids, topk.topk_weights, layer, scales, variant
        )
        row[variant] = metrics(ref, out)
        row[variant]["recv_counts_equal"] = torch.equal(ref_counts.cpu(), counts.cpu())
        references.append(ref.detach().clone())
    row["between_references"] = metrics(references[0], references[1])
    root = Path(os.environ["SGLANG_FUSEEP_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"rank{row['rank']}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("[REAL_MOE_AUDIT] " + json.dumps(row), flush=True)
    _seen.add(key)
