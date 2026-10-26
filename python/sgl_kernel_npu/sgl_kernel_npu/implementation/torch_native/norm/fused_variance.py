def fused_variance(x):
    # Preserve the kernel's input-precision multiply, FP32 reduction and output cast.
    return x.square().float().mean(dim=-1, keepdim=True).to(x.dtype)
