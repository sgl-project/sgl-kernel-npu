"""DSV4 NPU operators with registered production and torch_native backends."""

from .attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
    sparse_flash_mla,
    sparse_flash_mla_metadata,
)
from .cache import indexer_quant_cache, kv_compress_epilog
from .compressor import compressor
from .indexer import quant_lightning_indexer, quant_lightning_indexer_metadata
from .layout import NPU_KV_QUANT_PATH
from .mhc import mhc_post, mhc_pre
from .moe import moe_gating_top_k
from .rope import inplace_partial_rotary_mul

__all__ = [
    "sparse_flash_mla_metadata",
    "sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata",
    "mixed_quant_sparse_flash_mla",
    "quant_lightning_indexer_metadata",
    "quant_lightning_indexer",
    "kv_compress_epilog",
    "indexer_quant_cache",
    "mhc_pre",
    "mhc_post",
    "inplace_partial_rotary_mul",
    "compressor",
    "moe_gating_top_k",
    "NPU_KV_QUANT_PATH",
]
