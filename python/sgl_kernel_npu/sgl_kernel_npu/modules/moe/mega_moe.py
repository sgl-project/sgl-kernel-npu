"""Registered MegaMoE interface backed by CANN MC2."""

from typing import Any

from ...operators.utils.generation import Generation
from ...operators.utils.registry import get_impl_by_name, register_impl, register_operator

_mega_moe_fn = None
_get_symm_buffer_fn = None


def _load_backend():
    global _mega_moe_fn, _get_symm_buffer_fn
    if _mega_moe_fn is None or _get_symm_buffer_fn is None:
        from cann_ops_transformer.ops import (
            get_symm_buffer_for_mega_moe,
            mega_moe,
        )

        _mega_moe_fn = mega_moe
        _get_symm_buffer_fn = get_symm_buffer_for_mega_moe
    return _mega_moe_fn, _get_symm_buffer_fn


_SUPPORTED_GENERATIONS = frozenset({Generation.ARCH22, Generation.ARCH35})


@register_impl(
    "moe.mega_moe",
    "auto",
    supported_generations=_SUPPORTED_GENERATIONS,
)
def _cann_impl(*args: Any, **kwargs: Any):
    mega_moe_fn, _ = _load_backend()
    return mega_moe_fn(*args, **kwargs)


@register_impl(
    "moe.mega_moe",
    "cann",
    supported_generations=_SUPPORTED_GENERATIONS,
)
def _cann_named_impl(*args: Any, **kwargs: Any):
    return _cann_impl(*args, **kwargs)


@register_operator("moe.mega_moe")
def mega_moe(*args: Any, impl: str = "auto", **kwargs: Any):
    """Run MegaMoE with the same signature as the CANN provider."""

    return get_impl_by_name("moe.mega_moe", impl)(*args, **kwargs)


def get_symm_buffer_for_mega_moe(*args: Any, **kwargs: Any):
    """Create the persistent EP communication buffer for MegaMoE."""

    _, get_buffer = _load_backend()
    return get_buffer(*args, **kwargs)
