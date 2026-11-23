"""Fused Qwen prefill residual combine and branch RMS normalization."""
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("hyper_conn.hc_prefill_combine_norm", "auto", supported_generations={Generation.ARCH22})
def _triton(*args):
    from ...implementation.triton.hyper_conn.hc_prefill_combine_norm import hc_sp_combine_norm

    return hc_sp_combine_norm(*args)


@register_operator("hyper_conn.hc_prefill_combine_norm")
def hc_prefill_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count, *, impl="auto"):
    """Return the four combined residual streams and their normalized BF16 rows."""
    return get_impl_by_name("hyper_conn.hc_prefill_combine_norm", impl)(
        residual, block_output, injection_logits, norm_weight, eps, hc_count
    )
