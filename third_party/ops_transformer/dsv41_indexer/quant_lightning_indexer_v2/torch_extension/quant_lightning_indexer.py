# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0; see upstream LICENSE.
"""A3 CANN 9.2 binding: existing QLI name, three candidate modes."""

import torch
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl

QLI_METADATA_SIZE = 1024
QLI_METADATA_OP_NAME = "quant_lightning_indexer_metadata"


class QuantLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("quant_lightning_indexer", category="attention")

    def sources(self):
        return ["csrc/attention/quant_lightning_indexer.cpp"]

    def schema(self):
        return [
            "quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None,Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? causal_seqlens_kv_global=None, int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_k=None, str? layout_q=None, str? layout_k=None, int? mask_mode=None, int? cmp_ratio=None, bool? enable_dcp=None, int? cp_world=None, int? cp_rank=None, int? cp_kv_cache_interleave_size=None) -> Tensor",
            'quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1, str layout_q="BSND", str layout_k="BSND", int mask_mode=0, int cmp_ratio=1, int return_value=0, int candidate_mode=3, Tensor? candidate_topk_index=None, int candidate_topk_blocks=2048, int candidate_block_size=8) -> (Tensor, Tensor, Tensor)',
        ]

    def register_meta(self):
        @torch.library.register_fake(
            "cann_ops_transformer::quant_lightning_indexer_metadata"
        )
        def metadata_meta(*args, **kwargs):
            tensor = next(
                (x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)),
                None,
            )
            return torch.empty(
                1024,
                dtype=torch.int32,
                device=tensor.device if tensor is not None else "npu",
            )

        @torch.library.register_fake("cann_ops_transformer::quant_lightning_indexer")
        def compute_meta(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            topk,
            quant_mode,
            **kwargs
        ):
            shape = (*query.shape[:-2], 1, topk)
            candidates = (
                (*query.shape[:-2], 1, kwargs.get("candidate_topk_blocks", 2048))
                if kwargs.get("candidate_mode", 3) == 1
                else (0,)
            )
            return (
                torch.empty(shape, dtype=torch.int32, device=query.device),
                torch.empty(
                    shape if kwargs.get("return_value", 0) else (0,),
                    dtype=torch.float32 if kwargs.get("return_value", 0) else torch.bfloat16,
                    device=query.device,
                ),
                torch.empty(candidates, dtype=torch.int32, device=query.device),
            )


quant_lightning_indexer_op_builder = QuantLightningIndexerOpBuilder()
quant_lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), "quant_lightning_indexer", "PrivateUse1")
def _quant_lightning_indexer_impl(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_mode=3,
    candidate_topk_index=None,
    candidate_topk_blocks=2048,
    candidate_block_size=8
):
    module = quant_lightning_indexer_op_builder.load()
    common = (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )
    if return_value:
        if candidate_mode != 3:
            raise ValueError("Candidate modes do not return scores")
        indices, values = module.quant_lightning_indexer(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            topk,
            quant_mode,
            *common[:8],
            None,
            *common[8:],
            return_value,
            False,
            1,
            0,
            1
        )
        return indices, values, torch.empty(0, dtype=torch.int32, device=query.device)
    return module._quant_lightning_indexer_modes(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        candidate_topk_index,
        *common,
        candidate_mode,
        candidate_topk_blocks,
        candidate_block_size
    )


@impl(get_as_library(), QLI_METADATA_OP_NAME, "PrivateUse1")
def _quant_lightning_indexer_metadata_impl(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    cmp_residual_k: torch.Tensor | None = None,
    causal_seqlens_kv_global: torch.Tensor | None = None,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    layout_q: str | None = None,
    layout_k: str | None = None,
    mask_mode: int | None = None,
    cmp_ratio: int | None = None,
    enable_dcp: bool | None = None,
    cp_world: int | None = None,
    cp_rank: int | None = None,
    cp_kv_cache_interleave_size: int | None = None,
):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    enable_dcp = False if enable_dcp is None else enable_dcp
    cp_world = 1 if cp_world is None else cp_world
    cp_rank = 0 if cp_rank is None else cp_rank
    cp_kv_cache_interleave_size = (
        1 if cp_kv_cache_interleave_size is None else cp_kv_cache_interleave_size
    )

    op_module = quant_lightning_indexer_op_builder.load()
    return op_module.quant_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        causal_seqlens_kv_global,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        enable_dcp,
        cp_world,
        cp_rank,
        cp_kv_cache_interleave_size,
    )


@torch.library.register_kernel("cann_ops_transformer::" + QLI_METADATA_OP_NAME, None)
def quant_lightning_indexer_metadata_fallback(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    cmp_residual_k: torch.Tensor | None = None,
    causal_seqlens_kv_global: torch.Tensor | None = None,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    layout_q: str | None = None,
    layout_k: str | None = None,
    mask_mode: int | None = None,
    cmp_ratio: int | None = None,
    enable_dcp: bool | None = None,
    cp_world: int | None = None,
    cp_rank: int | None = None,
    cp_kv_cache_interleave_size: int | None = None,
):
    # 处理所有 tensor 都为 None 的情况
    # 调用 NPU 实现
    return _quant_lightning_indexer_metadata_impl(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        causal_seqlens_kv_global,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        enable_dcp,
        cp_world,
        cp_rank,
        cp_kv_cache_interleave_size,
    )


def quant_lightning_indexer(*args, **kwargs):
    return getattr(torch.ops, get_as_library().ns).quant_lightning_indexer(
        *args, **kwargs
    )


def quant_lightning_indexer_metadata(*args, **kwargs):
    return getattr(torch.ops, get_as_library().ns).quant_lightning_indexer_metadata(
        *args, **kwargs
    )
