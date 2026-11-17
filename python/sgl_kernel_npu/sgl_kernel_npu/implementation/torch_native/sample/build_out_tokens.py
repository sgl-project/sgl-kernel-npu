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
    correct_len = correct_len.to(torch.int64)
    keep = (
        torch.arange(gamma, device=draft_tokens.device)[None, :] < correct_len[:, None]
    )
    out = torch.zeros(
        (bs, verify_num_draft_tokens), dtype=torch.int64, device=draft_tokens.device
    )
    out[:, :gamma] = torch.where(keep, draft_tokens.to(torch.int64), 0)
    out.scatter_(1, correct_len.unsqueeze(1), bonus.to(torch.int64).unsqueeze(1))
    return out
