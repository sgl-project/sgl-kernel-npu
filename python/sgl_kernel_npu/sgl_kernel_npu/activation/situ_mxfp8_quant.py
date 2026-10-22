"""A5 AscendC SiTU + MXFP8 quantization."""

import torch


def situ_mxfp8_quant(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    beta: float = 4.0,
    linear_beta: float = 25.0,
):
    """Fuse Kimi-K3 SiTU and per-32-element MXFP8 quantization.

    ``hidden_states`` is the BF16 GMM1 output with shape ``[capacity, 6144]``.
    Only rows selected by ``group_list`` are processed. Returned scales have
    the same logical layout as ``npu_dynamic_mx_quant``: ``[capacity, 48, 2]``.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 6144:
        raise ValueError(
            "situ_mxfp8_quant expects hidden_states shape [capacity, 6144], "
            f"got {tuple(hidden_states.shape)}"
        )
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    return torch.ops.npu.situ_mxfp8_quant(
        hidden_states,
        group_list,
        group_list_type,
        float(beta),
        float(linear_beta),
    )
