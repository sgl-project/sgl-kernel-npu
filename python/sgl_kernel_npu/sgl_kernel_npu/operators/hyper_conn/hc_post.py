"""Hyper-connection post mixing backed by bundled cann-recipes-infer."""

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_recipes_fn = None
_native_fn = None


@register_impl("hyper_conn.hc_post", "auto")
def _auto_impl(x, residual, post, comb):
    version = detect_cann_version()
    if version != "9.1.0":
        raise RuntimeError(
            "hc_post auto supports CANN 9.1.0 only; detected CANN {}. "
            "Choose an explicit supported impl, such as torch_native.".format(version)
        )
    return get_impl_by_name("hyper_conn.hc_post", "cann_recipes_infer_cann910")(
        x, residual, post, comb
    )


@register_impl("hyper_conn.hc_post", "cann_recipes_infer_cann910")
def _recipes_impl(x, residual, post, comb):
    global _recipes_fn
    if _recipes_fn is None:
        from ..._third_party.sgl_cann_recipes_infer import npu_hc_post

        _recipes_fn = npu_hc_post
    return _recipes_fn(x, residual, post, comb)


@register_impl("hyper_conn.hc_post", "torch_native")
def _native_impl(x, residual, post, comb):
    global _native_fn
    if _native_fn is None:
        from ...implementation.torch_native.hyper_conn.hc_post import hc_post

        _native_fn = hc_post
    return _native_fn(x, residual, post, comb)


@register_operator("hyper_conn.hc_post")
def hc_post(x, residual, post, comb, *, impl="auto"):
    """Mix residual streams and add the weighted layer output.

    auto selects cann_recipes_infer_cann910 on CANN 9.1.0 (arch22/arch35)
    and raises on other releases. For stream j:
    y[j] = post[j] * x + sum_i(comb[i,j] * residual[i]).

    Args:
        x (Tensor): Layer output [T, D] or [B, S, D], fp16/bf16/fp32.
        residual (Tensor): Residual streams [T, 4, D] or [B, S, 4, D]; x dtype.
        post (Tensor): Output weights [T, 4] or [B, S, 4], fp16/bf16/fp32.
        comb (Tensor): Mixing weights [T, 4, 4] or [B, S, 4, 4]; post dtype.
        impl (str): auto, cann_recipes_infer_cann910, or torch_native.

    Inputs are contiguous and on the same device; D is 4096 or 7168.
    Returns a new tensor with residual's shape, dtype and device.
    """
    return get_impl_by_name("hyper_conn.hc_post", impl)(x, residual, post, comb)
