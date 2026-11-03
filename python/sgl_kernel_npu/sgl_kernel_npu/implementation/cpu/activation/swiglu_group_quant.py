from ...torch_native.activation.swiglu_group_quant import (
    swiglu_group_quant as _quantize,
)


def swiglu_group_quant(x, topk_weight, group_index, **kwargs):
    return _quantize(
        x.cpu(),
        None if topk_weight is None else topk_weight.cpu(),
        None if group_index is None else group_index.cpu(),
        **kwargs
    )
