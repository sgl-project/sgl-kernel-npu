"""Independent PyTorch implementation of the token-major convolution contract.

The calculation is expressed as history concatenation and per-tap products;
no CANN, legacy custom operator, or other implementation backend is called.
"""

import torch
import torch.nn.functional as F


def _present(tensor):
    return tensor is not None and tensor.numel() > 0


def _raw_span(tensor):
    # The old kernel reads from data_ptr without applying optional-tensor strides.
    # Keep the storage offset and do not normalize to a contiguous copy.
    return tensor.as_strided((tensor.numel(),), (1,))


@torch.no_grad()
def causal_conv1d(
    x,
    weight,
    conv_states,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
):
    """Compute y and mutate the caller's state, using FP32 accumulation.

    The oracle covers valid sequence windows, sufficient history capacity and
    distinct live cache slots. Concurrent requests sharing a cache slot have no
    established device ordering and are not a reference accuracy contract.
    Metadata reads synchronize when this backend runs on an accelerator.
    """
    # Match the old Host's required tensor, layout and dtype checks.
    if x.ndim not in (2, 3):
        raise ValueError("x must be 2D or 3D tensor")
    if weight.ndim != 2:
        raise ValueError("weight must be 2D tensor")
    if conv_states.ndim != 3:
        raise ValueError("conv_states must be 3D tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Only BF16 and FP16 are supported")
    if weight.dtype != x.dtype:
        raise ValueError("weight dtype must match x dtype")
    if conv_states.dtype != x.dtype:
        raise ValueError("conv_states dtype must match x dtype")
    for name, tensor in (("x", x), ("weight", weight), ("conv_states", conv_states)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    dim = x.shape[-1]
    width = weight.shape[0]
    if weight.shape[1] != dim:
        raise ValueError("weight.shape[1] must equal dim")
    if conv_states.shape[2] != dim:
        raise ValueError("conv_states.shape[2] must equal dim")
    if not 2 <= width <= 4:
        raise ValueError("Only support width in [2,4]")
    if x.ndim == 2 and (query_start_loc is None or query_start_loc.shape[0] < 2):
        raise ValueError("query_start_loc must have at least 2 elements")

    # Empty optional tensors have the same meaning as absent optional tensors.
    query = (
        _raw_span(query_start_loc.to(torch.int64))
        if _present(query_start_loc)
        else None
    )
    indices = (
        _raw_span(cache_indices.to(torch.int64)) if _present(cache_indices) else None
    )
    initial = (
        _raw_span(has_initial_state.to(torch.int64))
        if _present(has_initial_state)
        else None
    )
    accepted = (
        _raw_span(num_accepted_tokens.to(torch.int32))
        if _present(num_accepted_tokens)
        else None
    )
    weight_fp32 = weight.float()
    # The old Kernel reads bias bytes as x.dtype, without a Host bias cast.
    bias_fp32 = _raw_span(bias).view(x.dtype)[:dim].float() if _present(bias) else None
    batch = x.shape[0] if x.ndim == 3 else query_start_loc.shape[0] - 1
    state_len = conv_states.shape[1]
    history_len = width - 1
    is_update = run_mode == 1
    is_speculative = is_update and accepted is not None and width == 4

    # Snapshot before any write, keeping original raw BF16/FP16 history values.
    old_states = conv_states.clone()
    x_tokens = x.reshape(-1, dim)
    y = torch.empty_like(x)
    y_tokens = y.reshape(-1, dim)
    for seq in range(batch):
        if x.ndim == 3:
            start, length = seq * x.shape[1], x.shape[1]
        else:
            start = int(query[seq].item())
            length = int(query[seq + 1].item()) - start
        if length <= 0:
            continue
        cache = seq if indices is None else int(indices[seq].item())
        # The default seq->slot mapping does not consult pad_slot_id.
        if indices is not None and cache == pad_slot_id:
            continue

        has_history = is_update or (
            initial is not None and int(initial[seq].item()) != 0
        )
        offset = 0
        if is_speculative:
            offset = int(accepted[seq].item()) - 1
            if offset < 0:
                offset = 0
            elif offset > state_len - history_len:
                offset = state_len - history_len
        if has_history:
            history = old_states[cache, offset : offset + history_len]
        else:
            history = x.new_zeros((history_len, dim))
        tokens = x_tokens[start : start + length]
        extended_raw = torch.cat((history, tokens), dim=0)
        extended_fp32 = extended_raw.float()

        # Channel-wise convolution: each tap broadcasts over sequence tokens.
        acc = extended_fp32[:length] * weight_fp32[0]
        for tap in range(1, width):
            acc = acc + extended_fp32[tap : tap + length] * weight_fp32[tap]
        if bias_fp32 is not None:
            acc = acc + bias_fp32
        if activation_mode != 0:
            acc = F.silu(acc)
        y_tokens[start : start + length] = acc.to(x.dtype)

        # State stores raw inputs, never the convolved or activated output.
        if is_speculative and state_len >= 2 + length:
            conv_states[cache, :2] = old_states[cache, offset + 1 : offset + 3]
            conv_states[cache, 2 : 2 + length] = tokens
        else:
            conv_states[cache, :history_len] = extended_raw[-history_len:]
    return y
