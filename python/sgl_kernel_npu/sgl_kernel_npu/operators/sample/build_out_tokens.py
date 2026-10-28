"""Public build_out_tokens operator and backend registration."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("sample.build_out_tokens", "auto")
@register_impl("sample.build_out_tokens", "triton")
def _triton_impl(**kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.sample.build_out_tokens import (
            build_out_tokens_triton,
        )

        _triton_fn = build_out_tokens_triton
    return _triton_fn(**kwargs)


@register_impl("sample.build_out_tokens", "torch_native")
def _torch_native_impl(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.sample.build_out_tokens import (
            build_out_tokens_triton,
        )

        _torch_native_fn = build_out_tokens_triton
    return _torch_native_fn(**kwargs)


@register_operator("sample.build_out_tokens")
def build_out_tokens(
    *, draft_tokens, correct_len, bonus, verify_num_draft_tokens, gamma, impl="auto"
):
    """Dispatch to the selected build_out_tokens implementation."""
    return get_impl_by_name("sample.build_out_tokens", impl)(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=verify_num_draft_tokens,
        gamma=gamma,
    )
