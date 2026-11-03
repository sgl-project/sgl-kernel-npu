"""Shared ABI conversions for the old shared-KV and released MLA families."""

import torch

from ..utils._transformer_compat import (
    OperatorUnavailableError,
    attention_family,
    sparse_host_family,
)

HOST_LENGTH_KEYS = ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")


def without_host_lengths(kwargs):
    return {key: value for key, value in kwargs.items() if key not in HOST_LENGTH_KEYS}


def legacy_kwargs(kwargs, quantized, *, metadata=False):
    result = without_host_lengths(kwargs)
    result.pop("seqused_cmp_kv", None)
    result.pop("cmp_residual_kv", None)
    result["seqused_kv"] = result.pop("seqused_ori_kv", None)
    layout = result.get("layout_kv", "BSND")
    if layout == "PA_BBND":
        result["layout_kv"] = "PA_ND"
    elif layout not in ("BSND", "TND"):
        raise ValueError(f"Unsupported shared-KV layout: {layout}")
    if result.pop("topk_value_mode", 1) != 1:
        raise OperatorUnavailableError(
            "Legacy attention only supports logical compressed top-k indices"
        )
    if result.pop("cmp_topk_length", None) is not None:
        raise OperatorUnavailableError(
            "Legacy attention does not accept cmp_topk_length"
        )
    result.pop("ori_topk_length", None)
    result.pop("ori_topk", None)
    if result.get("ori_mask_mode", 4) == 0:
        # Dedicated native DSpark kernels select the explicit-index path by
        # tensor presence; their metadata still uses the SWA/window ABI.
        result["ori_mask_mode"] = 4
    if not result.get("has_cmp_kv", True) or result.get("cmp_ratio", 1) == 1:
        result["cmp_mask_mode"] = 3
    if quantized:
        if result.pop("quant_mode", 1) != 1:
            raise OperatorUnavailableError(
                "Legacy packed KV supports quant_mode=1 only"
            )
        result.update(kv_quant_mode=1, tile_size=64)
        if result.get("rope_head_dim", 64) != 64:
            raise OperatorUnavailableError("Legacy packed KV requires rope_head_dim=64")
    if metadata:
        # Only A5's host ABI has these optional scheduling bounds. The custom
        # device metadata entry has no corresponding parameters.
        result.pop("seqused_q", None)
    return result


def build_metadata(quantized, kwargs):
    family = attention_family(quantized)
    if not family.legacy:
        return family.ops[0](**without_host_lengths(kwargs))
    sparse = (
        kwargs.get("ori_mask_mode", 4) == 0
        or kwargs.get("ori_topk_length") is not None
        or kwargs.get("ori_topk", 0) > 0
    )
    if sparse and kwargs.get("has_cmp_kv", False):
        raise OperatorUnavailableError(
            "Legacy DSpark explicit original KV cannot be combined with compressed KV"
        )
    args = legacy_kwargs(kwargs, quantized, metadata=True)
    if sparse:
        family = sparse_host_family(quantized)
        for public, private in (
            ("cu_seqlens_q_cpu", "cu_seqlens_q"),
            ("seqused_ori_kv_cpu", "seqused_kv"),
        ):
            value = kwargs.get(public)
            if (
                value is None
                or value.device.type != "cpu"
                or value.dtype != torch.int32
            ):
                raise ValueError(
                    f"Legacy host metadata requires an existing CPU int32 {public}; no device copy is performed during capture"
                )
            count = kwargs["batch_size"] + (private == "cu_seqlens_q")
            args[private] = value[:count].contiguous()
        if not quantized:
            args.pop("max_seqlen_q", None)
            args.pop("max_seqlen_kv", None)
    return family.ops[0](**args)


def logical_to_physical(indices, lengths, q, table, cache, cu_q, layout):
    """Resolve original-KV indices for the native DSpark ABI using tensor ops."""
    if table is None:
        raise ValueError("Legacy sparse original KV requires ori_block_table")
    if layout == "TND":
        if cu_q is None:
            raise ValueError("TND sparse original KV requires cu_seqlens_q")
        token = torch.arange(q.shape[0], device=q.device)
        batch = (token[:, None] >= cu_q[None, 1:]).sum(-1).clamp(max=table.shape[0] - 1)
    elif layout == "BSND":
        batch = torch.arange(q.shape[0], device=q.device).repeat_interleave(q.shape[1])
    else:
        raise OperatorUnavailableError(
            "Legacy sparse indices support TND or BSND queries"
        )
    flat = indices.reshape(-1, indices.shape[-1]).long()
    valid = flat >= 0
    if lengths is not None:
        column = torch.arange(flat.shape[-1], device=q.device)
        valid = valid & (column < lengths.reshape(-1, 1))
    safe = flat.clamp(min=0, max=table.shape[1] * cache.shape[1] - 1)
    pages = table[batch[:, None], safe // cache.shape[1]].long()
    slots = pages * cache.shape[1] + safe % cache.shape[1]
    return torch.where(valid, slots, -1).to(torch.int32).reshape_as(indices)


def compute(quantized, q, kwargs):
    family = attention_family(quantized)
    if quantized:
        if kwargs.get("quant_mode", 1) != 1:
            raise OperatorUnavailableError(
                "The migrated paired KV reader/writer contract supports quant_mode=1"
            )
        width = 640 if family.legacy else 608
        for name in ("ori_kv", "cmp_kv"):
            cache = kwargs.get(name)
            if cache is not None and cache.shape[-1] != width:
                raise ValueError(f"{name} must use the selected {width}B KV encoding")
    if family.legacy:
        args = legacy_kwargs(kwargs, quantized)
        indices = kwargs.get("ori_sparse_indices")
        if kwargs.get("ori_mask_mode", 4) == 0 and indices is None:
            raise ValueError(
                "Legacy explicit sparse attention requires ori_sparse_indices"
            )
        if indices is not None:
            if kwargs.get("cmp_kv") is not None:
                raise OperatorUnavailableError(
                    "Legacy DSpark explicit original KV cannot be combined with compressed KV"
                )
            family = sparse_host_family(quantized)
            args["ori_sparse_indices"] = logical_to_physical(
                indices,
                kwargs.get("ori_topk_length"),
                q,
                kwargs.get("ori_block_table"),
                kwargs["ori_kv"],
                kwargs.get("cu_seqlens_q"),
                kwargs.get("layout_q", "BSND"),
            )
        args.pop("has_ori_kv", None)
        args.pop("has_cmp_kv", None)
        out, lse = family.ops[1](q, **args)
    else:
        # quant_mode is a keyword in both released 9.1 and 9.2 schemas.
        out, lse = family.ops[1](q, **without_host_lengths(kwargs))
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse
