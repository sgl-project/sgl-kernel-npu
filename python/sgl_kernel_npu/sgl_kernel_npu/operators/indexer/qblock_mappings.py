"""Public query-block mappings for sparse indexer prefill."""

from ...implementation.triton.indexer.flash_prefill_bnsd_score import (
    _build_qblock_mappings as build_qblock_mappings,
)

__all__ = ["build_qblock_mappings"]
