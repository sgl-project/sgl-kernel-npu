import torch


def speculative_state_scatter(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> torch.Tensor:
    dst_indices = dst_indices.to(torch.int32)
    src_indices = src_indices.to(torch.int32)
    step_indices = step_indices.to(torch.int32)
    valid = (dst_indices >= 0) & (src_indices >= 0) & (step_indices >= 0)
    requests = valid.nonzero(as_tuple=True)[0]
    if requests.numel() == 0:
        return dst

    dst_slots = dst_indices[requests].long()
    src_slots = src_indices[requests].long()
    steps = step_indices[requests].long()
    dst[:, dst_slots] = src[:, src_slots, steps]
    return dst
