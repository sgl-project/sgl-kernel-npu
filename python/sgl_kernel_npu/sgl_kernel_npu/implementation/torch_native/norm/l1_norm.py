def l1_norm(input):
    # This kernel normalizes by the signed sum, not the sum of absolute values.
    values = input.float()
    return values / values.sum(dim=-1, keepdim=True)
