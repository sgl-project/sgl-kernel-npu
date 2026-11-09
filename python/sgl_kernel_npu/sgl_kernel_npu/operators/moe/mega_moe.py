"""Registered MegaMoE interface backed by CANN MC2."""

from typing import Any

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_mega_moe_fn = None
_get_symm_buffer_fn = None


def _load_backend():
    global _mega_moe_fn, _get_symm_buffer_fn
    if _mega_moe_fn is None or _get_symm_buffer_fn is None:
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

        _mega_moe_fn = mega_moe
        _get_symm_buffer_fn = get_symm_buffer_for_mega_moe
    return _mega_moe_fn, _get_symm_buffer_fn


_SUPPORTED_GENERATIONS = frozenset({Generation.ARCH22, Generation.ARCH35})


@register_impl("moe.mega_moe", "auto", supported_generations=_SUPPORTED_GENERATIONS)
@register_impl("moe.mega_moe", "cann", supported_generations=_SUPPORTED_GENERATIONS)
def _cann_impl(*args: Any, **kwargs: Any):
    mega_moe_fn, _ = _load_backend()
    return mega_moe_fn(*args, **kwargs)


@register_operator("moe.mega_moe")
def mega_moe(*args: Any, impl: str = "auto", **kwargs: Any):
    """Dispatch tokens, run expert FFNs, and combine; auto selects CANN 9.2.

    Forward provider arguments unchanged: x [T,H], topk_ids/topk_weights [T,K],
    per-expert l1_weights/l2_weights and a live sym_buffer. Optional scales,
    biases, active mask and activation parameters follow the installed CANN
    provider. Return (output [T,H], local expert token counts). Tensor dtypes
    and layouts depend on architecture/quantization; see the bilingual API docs.
    The caller owns buffer lifetime and collective ordering on every EP rank.
    """

    return get_impl_by_name("moe.mega_moe", impl)(*args, **kwargs)


@register_impl(
    "moe.get_symm_buffer_for_mega_moe",
    "auto",
    supported_generations=_SUPPORTED_GENERATIONS,
)
@register_impl(
    "moe.get_symm_buffer_for_mega_moe",
    "cann",
    supported_generations=_SUPPORTED_GENERATIONS,
)
def _cann_buffer_impl(*args: Any, **kwargs: Any):
    _, get_buffer = _load_backend()
    return get_buffer(*args, **kwargs)


@register_operator("moe.get_symm_buffer_for_mega_moe")
def get_symm_buffer_for_mega_moe(*args: Any, impl: str = "auto", **kwargs: Any):
    """Create an EP communication buffer; auto selects the CANN 9.2 provider.

    Forward the process group and num_experts, num_max_tokens_per_rank,
    num_topk, hidden, intermediate_hidden and quantization options unchanged.
    All ranks must use compatible capacity/layout settings. The returned
    provider object is not cached here; synchronize work before destroy().
    """

    return get_impl_by_name("moe.get_symm_buffer_for_mega_moe", impl)(*args, **kwargs)
