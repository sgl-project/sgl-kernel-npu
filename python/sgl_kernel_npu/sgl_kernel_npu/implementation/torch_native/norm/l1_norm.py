import torch


def l1_norm(input):
    # Follow the original golden: normalize FP32 values with NumPy on the CPU.
    values = input.to(torch.float32).cpu().numpy()
    output = values / values.sum(axis=-1, keepdims=True)
    return torch.from_numpy(output).to(device=input.device)
