"""Compare dense-scatter and nonzero-scatter renormalization on one NPU."""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu
from sgl_kernel_npu.sample import probability


def dense_renorm(probs, values, indices):
    values.div_(values.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    return torch.zeros_like(probs).scatter_(-1, indices, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 10, 40])
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch_npu.npu.set_device(0)
    torch.manual_seed(137)
    optimized = probability._renorm_from_sorted_probs
    results = []
    for batch in args.batch_sizes:
        probs = torch.softmax(torch.randn((batch, args.vocab_size), device="npu"), -1)
        ks = torch.full(
            (batch,), min(20, args.vocab_size), device="npu", dtype=torch.long
        )
        ps = torch.full((batch,), 0.95, device="npu")
        for workload in ["helper_top20", "helper_dense", "top_k_then_top_p"]:
            values, indices = probs.sort(dim=-1, descending=True)
            if workload == "helper_top20":
                values[:, 20:] = 0

            def run():
                if workload == "top_k_then_top_p":
                    return probability.top_p_renorm_prob(
                        probability.top_k_renorm_prob(probs, ks), ps
                    )
                return probability._renorm_from_sorted_probs(
                    probs, values.clone(), indices
                )

            with patch.object(probability, "_renorm_from_sorted_probs", dense_renorm):
                expected = run()
            actual = run()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            methods = {
                "dense_scatter_ms": dense_renorm,
                "nonzero_scatter_ms": optimized,
            }
            timings = {name: [] for name in methods}
            for helper in methods.values():
                with patch.object(probability, "_renorm_from_sorted_probs", helper):
                    for _ in range(args.warmup):
                        run()
            for repeat in range(args.repeats):
                order = list(methods.items())
                if repeat % 2:
                    order.reverse()
                for name, helper in order:
                    with patch.object(probability, "_renorm_from_sorted_probs", helper):
                        torch_npu.npu.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            run()
                        torch_npu.npu.synchronize()
                    timings[name].append(
                        (time.perf_counter() - start) * 1000 / args.iterations
                    )
            row = {
                "batch": batch,
                "vocab": args.vocab_size,
                "workload": workload,
                "equal": True,
                "timings": timings,
                "medians_ms": {
                    name: statistics.median(v) for name, v in timings.items()
                },
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "device": torch_npu.npu.get_device_name(0),
                "args": vars(args) | {"output": str(args.output)},
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
