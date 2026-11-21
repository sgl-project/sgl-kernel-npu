import torch


def l1_norm(input, valid_rows=None):
    # Follow the original golden: normalize FP32 values with NumPy on the CPU.
    values = input.to(torch.float32).cpu().numpy()
    output = values / values.sum(axis=-1, keepdims=True)
    result = torch.from_numpy(output).to(device=input.device)
    if valid_rows is not None:
        rows = torch.arange(result.shape[0], device=result.device)
        result.masked_fill_((rows >= valid_rows).unsqueeze(1), 0)
    return result
