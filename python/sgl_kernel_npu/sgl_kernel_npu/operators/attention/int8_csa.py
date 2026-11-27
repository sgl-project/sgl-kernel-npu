"""CANN native INT8 compressed sparse attention for the signed 608-byte cache."""

from functools import lru_cache
from math import isclose

import torch

from ..utils.generation import Generation, get_hardware


@lru_cache(maxsize=1)
def _native_ops():
    if get_hardware().generation is not Generation.ARCH22:
        raise RuntimeError("INT8 CSA requires the CANN arch22 implementation.")
    try:
        # Import registers the private Torch namespace and its graph metadata.
        from ..._third_party import sgl_ops_transformer_int8_csa  # noqa: F401

        ops = torch.ops.sgl_ops_transformer_int8_csa
        return (
            ops.kv_compress_epilog,
            ops.mixed_quant_sparse_flash_mla_metadata,
            ops.mixed_quant_sparse_flash_mla,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "INT8 CSA requires sgl-kernel-npu built with the ops_transformer_int8_csa profile."
        ) from exc


def int8_csa_write(cache, x, slot_mapping, *, reserve_page_zero=False):
    """Write post-RoPE BF16 rows in place.

    A slot of -1 skips the row. With reserve_page_zero, all slots in the first
    page and out-of-range slots are skipped without changing the input shape.
    """
    if reserve_page_zero:
        x = x.reshape(-1, 512).contiguous()
        if x.shape[0] == 0:
            return
        slot_mapping = slot_mapping.reshape(-1)
        if x.shape[0] != slot_mapping.numel() or x.dtype != torch.bfloat16:
            raise ValueError("INT8 CSA requires one BF16 latent row per cache slot.")
        page_rows = cache.shape[1]
        live = (slot_mapping >= page_rows) & (slot_mapping < cache.shape[0] * page_rows)
        slot_mapping = torch.where(live, slot_mapping, -1).contiguous()

    writer, _, _ = _native_ops()
    writer(
        cache.view(torch.uint8),
        x,
        slot_mapping,
        quant_group_size=64,
        quant_mode="int8_bf16",
        round_scale=False,
        x_scale=1.0,
    )


def int8_csa_metadata(num_heads_q, num_heads_kv, head_dim, **kwargs):
    """Return the CANN scheduler descriptor for a batch of INT8 CSA queries."""
    _, metadata, _ = _native_ops()
    return metadata(num_heads_q, num_heads_kv, head_dim, 1, rope_head_dim=64, **kwargs)


def int8_csa_attention(q, *, sinks=None, **kwargs):
    """Read both histories as signed INT8 records and return attention and LSE."""
    _, _, attention = _native_ops()
    if sinks is None:
        sinks = torch.full(
            (q.shape[1],), -torch.inf, dtype=torch.float32, device=q.device
        )
    else:
        sinks = sinks.reshape(-1).float().contiguous()
    return attention(q, 1, rope_head_dim=64, topk_value_mode=1, sinks=sinks, **kwargs)


@lru_cache(maxsize=None)
def check_int8_csa_support(device):
    """Verify the native 608-byte A16C8 contract on an Atlas A3 device."""
    import acl

    soc_name = acl.get_soc_name()
    if not soc_name.startswith("Ascend910_93"):
        raise ValueError(f"INT8 CSA requires Atlas A3, got {soc_name}.")

    try:
        win = torch.zeros((2, 16, 1, 608), dtype=torch.int8, device=device)
        cmp = torch.zeros_like(win)
        q = torch.zeros((1, 1, 512), dtype=torch.bfloat16, device=device)
        kv = torch.zeros_like(q).view(1, 512)
        kv[:, -1] = 2
        slot = torch.tensor([16], dtype=torch.int32, device=device)
        lens = torch.ones(1, dtype=torch.int32, device=device)
        common = dict(
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seqused_ori_kv=lens,
            seqused_cmp_kv=lens,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
        int8_csa_write(win, kv, slot)
        int8_csa_write(cmp, kv, slot)
        page_table = lens.view(1, 1)
        metadata = int8_csa_metadata(
            1,
            1,
            512,
            batch_size=1,
            cmp_topk=16,
            has_ori_kv=True,
            has_cmp_kv=True,
            **common,
        )
        indices = torch.full((1, 1, 16), -1, dtype=torch.int32, device=device)
        indices[..., 0] = 0
        out, _ = int8_csa_attention(
            q,
            ori_kv=win,
            cmp_kv=cmp,
            cmp_sparse_indices=indices,
            ori_block_table=page_table,
            cmp_block_table=page_table,
            metadata=metadata,
            sinks=torch.zeros(1, dtype=torch.float32, device=device),
            softmax_scale=512**-0.5,
            **common,
        )
        # Two value-2 rows and one sink have equal logits and yield 4/3.
        if out.dtype != torch.bfloat16 or not isclose(
            out[0, 0, -1].item(), 4 / 3, abs_tol=0.02
        ):
            raise RuntimeError(
                "The native INT8 CSA startup probe returned an invalid result."
            )
    except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"INT8 CSA failed the native CANN probe on {soc_name}. "
            "Use a matching sgl-kernel-npu wheel with the ops_transformer_int8_csa profile."
        ) from exc
