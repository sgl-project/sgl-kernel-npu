"""Public fused PTO GDN pipeline through the standalone CANN build."""

from ...utils.generation import Generation
from ...utils.registry import get_impl_by_name, register_impl, register_operator

_ascendc_fn = None
_torch_native_fn = None


@register_impl(
    "attention.gdn.mega_chunk_gdn", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.gdn.mega_chunk_gdn", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(*args, **kwargs):
    global _ascendc_fn
    if _ascendc_fn is None:
        from ._mega_chunk_gdn import run_mega_chunk_gdn

        _ascendc_fn = run_mega_chunk_gdn
    return _ascendc_fn(*args, **kwargs)


@register_impl("attention.gdn.mega_chunk_gdn", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.mega_chunk_gdn import (
            run_mega_chunk_gdn,
        )

        _torch_native_fn = run_mega_chunk_gdn
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.gdn.mega_chunk_gdn")
def mega_chunk_gdn(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    *,
    impl="auto",
):
    """Return (g_sum, o, A_inv, final_state, w, h, v_new); auto selects ascendc.

    The native implementation retains the original PTO pipeline and helper.
    Inputs use packed B=1: q/k [1,T,Hg,128], v [1,T,H,128], g/beta
    [1,T,H]. H is one of 8/12/16/24/32/48/64 and divisible by Hg.
    Q/K/V/beta and initial_state are converted to FP16 by the copied helper.
    g is FP32. cu_seqlens is converted to INT32; None means one sequence.
    Chunks have 128 tokens; states use [N,H,128,128]. No Q/K normalization
    or input guard is added here. Optional final_state is FP32.
    """
    return get_impl_by_name("attention.gdn.mega_chunk_gdn", impl)(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
    )
