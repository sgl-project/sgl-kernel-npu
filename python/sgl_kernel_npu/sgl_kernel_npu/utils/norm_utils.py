"""Shared gate-activation helpers for gated LayerNorm and RMSNorm."""

# Gate activations implemented by the NPU gated norm kernels. Keep in sync with
# the reference implementation in sglang's layernorm_gated.py.
SUPPORTED_GATE_ACTIVATIONS = ("swish", "silu", "sigmoid")


def validate_gate_activation(activation):
    """Normalize None to 'swish' and reject unsupported gate activations."""
    if activation is None:
        return "swish"
    if activation not in SUPPORTED_GATE_ACTIVATIONS:
        raise ValueError(
            f"Unsupported gated RMSNorm activation on NPU: {activation!r}; "
            f"expected one of {SUPPORTED_GATE_ACTIVATIONS}"
        )
    return activation
