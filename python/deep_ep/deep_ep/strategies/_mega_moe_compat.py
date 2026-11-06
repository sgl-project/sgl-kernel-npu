"""Compatibility with MegaMoE operators predating activation_params."""

import inspect
from functools import wraps


def adapt_mega_moe_operator(operator):
    signature = inspect.signature(operator)
    parameters = signature.parameters
    if "activation_params" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return operator
    if "activation" not in parameters:
        raise RuntimeError("MegaMoE operator lacks activation selection")

    @wraps(operator)
    def legacy_swiglu(*args, **kwargs):
        activation_params = kwargs.pop("activation_params", None)
        activation = kwargs.get("activation", inspect.Parameter.empty)
        if activation is inspect.Parameter.empty:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            activation = bound.arguments["activation"]
        if activation != "swiglu" or activation_params is not None:
            raise ValueError(
                "Legacy MegaMoE supports only swiglu with activation_params=None; "
                "install a matching operator for non-default activation parameters"
            )
        return operator(*args, **kwargs)

    # Retain a truthful signature for callers validating the dependency contract.
    legacy_swiglu.__signature__ = signature.replace(
        parameters=[
            *parameters.values(),
            inspect.Parameter(
                "activation_params", inspect.Parameter.KEYWORD_ONLY, default=None
            ),
        ]
    )
    return legacy_swiglu
