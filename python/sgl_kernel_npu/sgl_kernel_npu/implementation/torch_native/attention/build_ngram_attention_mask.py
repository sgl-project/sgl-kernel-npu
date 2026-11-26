import torch


def build_ngram_attention_mask(tree_mask, prefix_lens, draft_token_num, padded_kv_len):
    batch, d = prefix_lens.numel(), draft_token_num
    if d == 0:
        return torch.empty(
            (batch, 1, 0, padded_kv_len), dtype=torch.bool, device=tree_mask.device
        )
    keys = torch.arange(padded_kv_len, device=tree_mask.device).view(1, 1, -1)
    prefix = prefix_lens.to(tree_mask.device).reshape(batch, 1, 1)
    draft = keys - prefix
    indices = draft.clamp(0, d - 1).expand(batch, d, padded_kv_len).long()
    ancestors = tree_mask.reshape(batch, d, d).bool().gather(2, indices)
    return ((keys >= prefix) & ~((draft < d) & ancestors)).unsqueeze(1)
