"""Profile steady-state DSpark sampling kernels against the NPU Torch paths.

Run under msprof, one (op, batch) per output directory. Each implementation
runs five warmups then twenty measured iterations. In op_summary, retain the
last twenty occurrences of each participating task to exclude compilation.
"""

import argparse
import json

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from sgl_kernel_npu.sample import (
    compute_dflash_correct_drafts_and_bonus,
    markov_greedy_step_triton,
    row_argmax,
    softmax_temp_triton,
)


def make_case(op, batch, vocab, rank, block):
    if op == "markov":
        base = torch.randn(batch, vocab, device="npu", dtype=torch.bfloat16)
        embeds = torch.randn(batch, rank, device="npu", dtype=torch.bfloat16)
        weight = torch.randn(vocab, rank, device="npu", dtype=torch.bfloat16)
        return (
            lambda: (base + F.linear(embeds, weight)).argmax(-1),
            lambda: markov_greedy_step_triton(
                base_logits=base, prev_embeds=embeds, w2_weight=weight
            ),
        )
    if op == "argmax":
        logits = torch.randn(batch * block, vocab, device="npu", dtype=torch.float32)
        return lambda: logits.argmax(-1), lambda: row_argmax(logits)
    if op == "accept":
        target = torch.randint(vocab, (batch, block), device="npu")
        candidates = torch.cat((target[:, :1], target[:, :-1]), dim=1)

        def reference():
            lengths = (candidates[:, 1:] == target[:, :-1]).int().cumprod(1).sum(1)
            bonus = target[torch.arange(batch, device="npu"), lengths]
            return lengths.int(), bonus.long()

        return reference, lambda: compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=target
        )
    logits = torch.randn(batch * block, vocab, device="npu", dtype=torch.float32)
    temperatures = torch.full((batch, 1), 0.7, device="npu")

    def reference():
        temps = torch.repeat_interleave(temperatures.reshape(batch).float(), block)
        return torch.softmax(logits.float() / temps[:, None], dim=-1)

    return reference, lambda: softmax_temp_triton(
        logits=logits, temperatures=temperatures, rows_per_request=block
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op", choices=("markov", "argmax", "accept", "softmax"), required=True
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--block", type=int, default=9)
    args = parser.parse_args()
    torch.manual_seed(42)
    reference, kernel = make_case(
        args.op, args.batch, args.vocab, args.rank, args.block
    )
    for fn in (reference, kernel):
        for _ in range(5):
            result = fn()
        torch.npu.synchronize()
    for fn in (reference, kernel):
        for _ in range(20):
            result = fn()
        torch.npu.synchronize()
    print(
        json.dumps(
            {
                **vars(args),
                "warmup": 5,
                "iterations": 20,
                "device": torch.npu.get_device_name(),
            }
        )
    )
    del result


if __name__ == "__main__":
    main()
