"""Model-specific NPU PLE convolution; state handling and SiLU stay outside."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(None)
def _vector_cores(device_index):
    cores = triton.runtime.driver.active.utils.get_device_properties(device_index).get('num_vectorcore', 0)
    if cores <= 0:
        raise RuntimeError('NPU vector core count is unavailable')
    return cores


def _validate(x, weight, dilation, single_token):
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError('Expected input [B,10240,T] and weight [10240,1,4]')
    if x.shape[1] != 10240 or tuple(weight.shape) != (10240, 1, 4):
        raise ValueError('PLE requires C=10240 and weight [10240,1,4]')
    if x.device != weight.device or x.device.type != 'npu':
        raise ValueError('Input and weight must be on the same NPU')
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        raise ValueError('Input and weight must have the same BF16 or FP16 dtype')
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('Input and weight must be contiguous')
    if not isinstance(dilation, int) or isinstance(dilation, bool) or dilation != 3:
        raise ValueError('PLE requires integer dilation=3')
    if not isinstance(single_token, bool):
        raise ValueError('single_token must be a bool')
    width = x.shape[2] - 9
    if width not in (1, 2, 3, 4) or (single_token and width != 1):
        raise ValueError('PLE requires W in 1..4; single_token=True requires W=1')
    return width


def launch_info(x, weight, dilation, *, single_token=False):
    """Check the model contract and choose settings for one convolution kernel.

    What this computes
    ------------------
    Like the GPU model's F.conv1d call, each channel uses its own K weights;
    channels are not mixed. The operation has stride=1, groups=C, no bias
    and no additional padding. State preparation and SiLU stay in the caller.

    Inputs and axes
    ---------------
        conv_input   [B, C, T]   cached history followed by new token values
        weight       [C, 1, K]   one filter per channel; keep the middle axis

        B   batch size, including any rows padded by the caller
        C   channels: hidden_size(2560) * hc_count(4) = 10240
        K   filter size: 4
        T   history length + number of new tokens
        W   output width: T - (K - 1) * dilation = T - 9

    Qwen4ExpPLELayer fixes C and K as above and uses dilation=ngram_size=3,
    giving 9 history positions. C is not divided by tensor-parallel size.
    B is dynamic, including zero; benchmark batch sizes are not a limit.

    What the caller must provide
    ----------------------------
    - Concatenate the 9 history positions and W new positions along T.
      For example: cat([state, new_tokens_in_BCT_layout], dim=-1).
    - Pass contiguous input and weight tensors on the same NPU, with the
      same dtype: BF16 or FP16. Keep the model's weight.to(input.dtype)
      before this call, and pass the full [C, 1, K] weight without squeezing.
      The wrapper does not cast mismatched dtypes or repair tensor layouts.
    - Use dilation=3 and W in {1, 2, 3, 4}. For current topk=1 verification,
      W=steps+1: steps 1/2/3 produce W=2/3/4. The inspected model's QSA
      compression ratio limits verification to 4. Ordinary prefill is
      outside this wrapper's integration scope.
    - Pass a bool for single_token; True is valid only when W=1.
      Tensor spans must also pass the int32 safety check below.

    The caller retains state/tracking writes, MTP caching, padding and valid
    token handling, output-token selection and SiLU. This kernel computes
    every supplied row; it does not decide which rows/tokens are valid.

    Outputs and examples
    --------------------
        single_token=True    [B, C]      requires W=1
        single_token=False   [B, W, C]   token axis before channel axis

        Input shape       W   single_token   Output shape
        [B, 10240, 10]    1   True           [B, 10240]
        [B, 10240, 10]    1   False          [B, 1, 10240]
        [B, 10240, 11]    2   False          [B, 2, 10240]
        [B, 10240, 12]    3   False          [B, 3, 10240]
        [B, 10240, 13]    4   False          [B, 4, 10240]

    All examples use weight [10240, 1, 4] and dilation=3. Outputs are
    contiguous and preserve input dtype/device. Inputs and weights are
    not modified. Unlike F.conv1d's [B, C, W] result, the general output
    is already arranged as [B, W, C] for the model.

    One computation path
    --------------------
    Every supported nonempty input uses _packed_conv, for both dtypes and
    every supported W. B=0 returns an empty output after validation without
    launching a kernel. Unsupported metadata raises; there is no fallback.

    Internal scheduling (the same algorithm in all cases)
    ----------------------------------------------------
    A tile is a group of channels processed together. Tile sizes are tuning
    choices, not different convolution implementations:

        Output width   Channel tile for B=1   Channel tile for B>1
        W=1            512                    1024
        W=2..4         256                     512

    These tile sizes divide C evenly. Each tile loads input values once,
    forms products and sums them in FP32, then stores in the input dtype.
    Power-of-two loads mask their unused tail. The grid uses at most the
    vector core count, with each program looping over its assigned tiles.
    Programs may reuse weights between batches when their channel tile
    stays the same; no weights are cached across wrapper calls.
    """
    width = _validate(x, weight, dilation, single_token)
    b, c, t = x.shape
    block = (512 if b == 1 else 1024) if width == 1 else (256 if b == 1 else 512)
    # Input span plus the padded tile load bounds all relative int32 offsets.
    # Output span B*W*C is smaller; storage offsets are part of the base pointer.
    if b*c*t + triton.next_power_of_2(block*t) >= 2**31:
        raise ValueError('PLE input span plus tile margin exceeds safe int32 indexing')
    tasks = b * (c // block)
    return dict(branch='empty' if b == 0 else 'triton', output_width=width,
                block_size=block if b else None, tasks=tasks,
                grid=min(tasks, _vector_cores(x.device.index)) if b else 0,
                index_bits=32 if b else None)


@triton.jit
def _packed_conv(X, Weight, Out, C: tl.constexpr, T: tl.constexpr, W: tl.constexpr,
                 K: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                 LOAD: tl.constexpr, TASKS: tl.constexpr, GRID: tl.constexpr):
    channel = tl.arange(0, BLOCK)
    iw = tl.arange(0, BLOCK*K)
    # Reuse is safe only if this program visits the same channels in each batch.
    REUSE: tl.constexpr = TASKS > GRID and GRID % tl.cdiv(C, BLOCK) == 0
    if REUSE:
        c0 = (tl.program_id(0) % tl.cdiv(C, BLOCK)) * BLOCK
        w = tl.load(Weight + c0*K+iw)
        w0 = tl.gather(w, channel*K, 0).to(tl.float32)
        w1 = tl.gather(w, channel*K+1, 0).to(tl.float32)
        w2 = tl.gather(w, channel*K+2, 0).to(tl.float32)
        w3 = tl.gather(w, channel*K+3, 0).to(tl.float32)
    for task in range(tl.program_id(0), TASKS, GRID):
        # One task covers one batch row and one consecutive group of channels.
        b = task // tl.cdiv(C, BLOCK)
        c0 = (task % tl.cdiv(C, BLOCK)) * BLOCK
        ix = tl.arange(0, LOAD)
        x = tl.load(X + b * C * T + c0 * T + ix,
                    (ix < BLOCK*T) & (c0*T+ix < C*T), other=0)
        if not REUSE:
            w = tl.load(Weight + c0*K+iw, c0*K+iw < C*K, other=0).to(X.dtype.element_ty)
        for token in tl.static_range(W):
            acc = tl.full((BLOCK,), 0, tl.float32)
            for tap in tl.static_range(K):
                # For output token t, read input positions t, t+3, t+6, t+9.
                a = tl.gather(x, channel*T+token+tap*D, 0)
                if REUSE:
                    if tap == 0:
                        v = w0
                    elif tap == 1:
                        v = w1
                    elif tap == 2:
                        v = w2
                    else:
                        v = w3
                else:
                    v = tl.gather(w, channel*K+tap, 0)
                prod = a.to(tl.float32)*v.to(tl.float32)
                acc = acc + prod
            # Write directly in the model's [batch, token, channel] order.
            tl.store(Out + (b*W+token)*C + c0 + channel, acc)


def short_conv(conv_input, weight, dilation, *, single_token=False):
    """Compute only the model convolution; see launch_info for the full contract."""
    info = launch_info(conv_input, weight, dilation, single_token=single_token)
    width = info['output_width']
    b, c, t = conv_input.shape
    out = torch.empty((b, width, c), dtype=conv_input.dtype, device=conv_input.device)
    if b:
        block = info['block_size']
        _packed_conv[(info['grid'],)](
            conv_input, weight, out, c, t, width, 4, dilation,
            block, triton.next_power_of_2(block*t), info['tasks'], info['grid'],
            enable_fp_fusion=False)
    return out.squeeze(1) if single_token else out
