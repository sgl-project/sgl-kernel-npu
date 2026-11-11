"""Native NPU convolution reference. Run outside graph capture."""

import torch.nn.functional as F


def short_conv(conv_input, weight, dilation, *, single_token=False):
    batch, channels, time = conv_input.shape
    width = time - 9
    if batch == 0:
        return conv_input.new_empty(
            (0, channels) if single_token else (0, width, channels)
        )
    result = F.conv1d(conv_input, weight, dilation=dilation, groups=channels)
    return result.squeeze(-1) if single_token else result.transpose(1, 2).contiguous()
