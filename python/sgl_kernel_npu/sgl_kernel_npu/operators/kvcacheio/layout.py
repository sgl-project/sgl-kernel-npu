"""Shared mixed-quant KV reader/writer contract for DSV4 on A5."""

import math
from typing import NamedTuple


class NpuKvQuantPath(NamedTuple):
    quant_group_size: int
    scale_itemsize: int
    row_alignment: int
    reader_quant_mode: int
    writer_quant_mode: str = "fp8_bf16"

    def packed_dim(self, nope_dim, rope_dim):
        size = (
            nope_dim
            + 2 * rope_dim
            + math.ceil(nope_dim / self.quant_group_size) * self.scale_itemsize
        )
        return math.ceil(size / self.row_alignment) * self.row_alignment


NPU_KV_QUANT_PATH = NpuKvQuantPath(64, 2, 32, 1)
LEGACY_NPU_KV_QUANT_PATH = NpuKvQuantPath(64, 1, 128, 1, "fp8_e8m0")


def get_npu_kv_quant_path():
    """Choose the immutable reader/writer contract before pool allocation."""
    from ..utils._transformer_compat import attention_family

    return (
        LEGACY_NPU_KV_QUANT_PATH if attention_family(True).legacy else NPU_KV_QUANT_PATH
    )
