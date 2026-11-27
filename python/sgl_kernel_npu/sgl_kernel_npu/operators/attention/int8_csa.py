"""CANN native INT8 compressed sparse attention for the signed 608-byte cache."""

from functools import lru_cache

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


def int8_csa_write(cache, x, slot_mapping):
    """Write post-RoPE BF16 rows in place. A slot of -1 skips the row."""
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


def int8_csa_attention(q, **kwargs):
    """Read both histories as signed INT8 records and return attention and LSE."""
    _, _, attention = _native_ops()
    return attention(q, 1, rope_head_dim=64, topk_value_mode=1, **kwargs)
