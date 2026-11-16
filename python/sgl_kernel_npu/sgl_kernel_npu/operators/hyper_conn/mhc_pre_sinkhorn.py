"""Inference mHC projection and Sinkhorn normalization."""

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl("hyper_conn.mhc_pre_sinkhorn", "auto")
def _auto_impl(*args):
    version = detect_cann_version()
    if version.startswith("9.1."):
        backend = "ops_transformer_cann910"
    elif version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("hyper_conn.mhc_pre_sinkhorn", backend)(*args)


def _run_vendor(fn, x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
    # Both released wrappers accept BSND and return a flattened residual matrix.
    y, post, comb = fn(
        x.unsqueeze(1), phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps
    )
    return y.squeeze(1), post.squeeze(1), comb.squeeze(1)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "ops_transformer_cann910")
def _cann910_impl(*args):
    version = detect_cann_version()
    if not version.startswith("9.1."):
        raise RuntimeError(f"This implementation requires CANN 9.1.x, got {version}")
    from ..._third_party.sgl_ops_transformer_cann910 import ops

    return _run_vendor(ops.mhc_pre_sinkhorn, *args)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "cann_ops_transformers_cann920")
def _cann920_impl(*args):
    version = detect_cann_version()
    if not version.startswith("9.2."):
        raise RuntimeError(f"This implementation requires CANN 9.2.x, got {version}")
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    return _run_vendor(ops.mhc_pre_sinkhorn, *args)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "cann_ops_transformer")
def _cann_impl(*args):
    """Retain the explicitly requested system-installed implementation."""
    global _cann_fn
    if _cann_fn is None:
        from cann_ops_transformer.ops import mhc_pre_sinkhorn

        _cann_fn = mhc_pre_sinkhorn
    return _run_vendor(_cann_fn, *args)


@register_impl("hyper_conn.mhc_pre_sinkhorn", "torch_native")
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.mhc_pre_sinkhorn import (
            mhc_pre_sinkhorn,
        )

        _torch_native_fn = mhc_pre_sinkhorn
    return _torch_native_fn(*args)


@register_operator("hyper_conn.mhc_pre_sinkhorn")
def mhc_pre_sinkhorn(
    x,
    phi,
    alpha,
    bias,
    hc_mult=4,
    num_iters=20,
    hc_eps=1e-6,
    norm_eps=1e-6,
    *,
    impl="auto",
):
    """Project [T,N,D] streams with the bundled CANN release backend.

    phi has shape [(2+N)*N,N*D], alpha [3], and bias [(2+N)*N].
    Returns (hin [T,D], hpost [T,N], hres [T,N*N]), preserving vendor
    output dtypes. Input constraints are delegated to the selected backend.
    No output layer normalization is fused.
    """
    impl_fn = get_impl_by_name("hyper_conn.mhc_pre_sinkhorn", impl)
    return impl_fn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps)
