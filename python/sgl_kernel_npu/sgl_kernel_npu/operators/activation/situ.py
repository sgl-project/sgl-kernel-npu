"""Public situ."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("activation.situ", "auto")
@register_impl("activation.situ", "triton")
def _triton_impl(
    hidden_states,
    group_list,
    group_list_type,
    *,
    need_quant,
    beta=4.0,
    linear_beta=25.0
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.situ import situ

        _triton_fn = situ
    return _triton_fn(
        hidden_states,
        group_list,
        group_list_type,
        need_quant=need_quant,
        beta=beta,
        linear_beta=linear_beta,
    )


@register_impl("activation.situ", "torch_native")
def _torch_native_impl(
    hidden_states,
    group_list,
    group_list_type,
    *,
    need_quant,
    beta=4.0,
    linear_beta=25.0
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.situ import situ

        _torch_native_fn = situ
    return _torch_native_fn(
        hidden_states,
        group_list,
        group_list_type,
        need_quant=need_quant,
        beta=beta,
        linear_beta=linear_beta,
    )


@register_operator("activation.situ")
def situ(
    hidden_states,
    group_list,
    group_list_type,
    *,
    need_quant,
    beta=4.0,
    linear_beta=25.0,
    impl="auto"
):
    """Grouped SiTU applies beta*tanh(gate/beta)*sigmoid(gate) to the first half of each row and multiplies by the second half, optionally bounded by linear_beta*tanh(up/linear_beta). Only routed rows are written. Optional INT8 output uses per-row max-absolute scales, a 1e-30 scale floor, floor(x/scale+0.5), and saturation to [-128,127].

    Args:
        hidden_states (Tensor): Contiguous floating [rows,2*D] input. Default: Required.
        group_list (Tensor): Contiguous INT32/INT64 expert counts. Type 0 requires an extra terminal total at storage offset numel(), outside the logical view. Default: Required.
        group_list_type (int): 1 for counts; 0 for cumulative metadata with terminal storage. Default: Required.
        need_quant (bool): Return INT8 values and FP32 row scales when true. Default: Required.
        beta (float): Softplus multiplier or SiTU saturation parameter, according to the operation. Default: 4.0.
        linear_beta (float or None): SiTU up-channel bound; None disables the tanh bound. Default: 25.0.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [rows,D] activation or INT8 output and [rows] FP32 scales (None without quantization).
    """
    return get_impl_by_name("activation.situ", impl)(
        hidden_states,
        group_list,
        group_list_type,
        need_quant=need_quant,
        beta=beta,
        linear_beta=linear_beta,
    )
