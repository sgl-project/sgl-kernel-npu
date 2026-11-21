"""A3 Qwen short-prefill sparse attention with 128-token KV pages."""
from functools import lru_cache

import torch
from torch_npu.npu._backends import get_soc_version

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

# The SDK groups SoC revisions by tens. A2 and A3 share ARCH22, but the
# bundled Cube operator has a compiled kernel only for the A3 family.
_A3_SOC_FAMILY = 250


@lru_cache(maxsize=1)
def is_qsa_cube_attention_supported():
    """Return whether the installed kernel targets this device."""
    if get_hardware().generation is not Generation.ARCH22:
        return False
    soc = get_soc_version()
    return soc // 10 * 10 == _A3_SOC_FAMILY


@register_impl("attention.qsa_cube_attention", "auto", supported_generations={Generation.ARCH22})
def _cube(q, k, v, indices, pages, requests):
    if not is_qsa_cube_attention_supported():
        raise RuntimeError("The short QSA Cube kernel requires A3")
    return torch.ops.npu.sgl_qsa_cube_attention(q, k, v, indices, pages, requests)


@register_operator("attention.qsa_cube_attention")
def qsa_cube_attention(q, k, v, indices, pages, requests, *, impl="auto"):
    """Compute BF16 attention for 2..127 query rows with three D256 heads on A3.

    K/V are [physical pages, 128, 1, 256]. Each contiguous int32 index row holds
    either 2051 logical token slots padded with -1, or 512 four-token group IDs
    followed by the query position and request length. The int32 request IDs
    select rows of the logical-to-physical page table. Scale is fixed at 1/16.
    """
    return get_impl_by_name("attention.qsa_cube_attention", impl)(q, k, v, indices, pages, requests)
