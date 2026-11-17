import torch.nn.functional as F


def markov_greedy_step_triton(*, base_logits, prev_embeds, w2_weight):
    return (base_logits + F.linear(prev_embeds, w2_weight)).argmax(dim=-1)
