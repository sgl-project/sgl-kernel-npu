import deep_ep
import torch
from utils import calc_diff, per_token_cast_back


def normal_test(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    buffer: deep_ep.Buffer,
):
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    buffer_size = 256
    config = deep_ep.Config(24, 8, buffer_size)

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }

    (
        recv_x,
        _,
        _,
        _,
        handle,
        _,
    ) = buffer.dispatch(**dispatch_args)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        topk_weights_recv,
    ) = handle
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    combine_args = {
        "x": recv_x,
        "handle": handle,
        "config": config,
        "async_finish": False,
        "topk_weights": topk_weights_recv,
    }
    (
        combined_x,
        _,
        _,
    ) = buffer.combine(**combine_args)

    assert (
        calc_diff(
            combined_x.float(),
            x * topk_weights_recv.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
        )
        < 5e-5
    )
