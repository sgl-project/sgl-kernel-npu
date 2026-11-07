import torch


def build_out_tokens_triton(
    *, draft_tokens, correct_len, bonus, verify_num_draft_tokens, gamma
):
    if gamma < 0 or verify_num_draft_tokens != gamma + 1:
        raise ValueError("verify_num_draft_tokens must equal gamma + 1")
    if draft_tokens.ndim != 2 or draft_tokens.shape[1] != gamma:
        raise ValueError("draft_tokens must have shape [batch, gamma]")
    bs = draft_tokens.shape[0]
    if correct_len.shape != (bs,) or bonus.shape != (bs,):
        raise ValueError("correct_len and bonus must have shape [batch]")
    out = torch.cat(
        (
            draft_tokens.to(torch.int64),
            draft_tokens.new_zeros((bs, 1), dtype=torch.int64),
        ),
        dim=1,
    )
    out.scatter_(
        1, correct_len.to(torch.int64).unsqueeze(1), bonus.to(torch.int64).unsqueeze(1)
    )
    return out
