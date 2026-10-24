"""Compare Philox sampling with precomputed Exp(1) noise on NPU.

Both eager and graph replay timings are reported. Graphs are measurement tools;
the sampler itself never captures a graph. Random generation is timed separately
and together with selection so that moving it outside cannot hide its cost.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_sampler(source):
    if source is None:
        from sgl_kernel_npu.sample import markov_sample_step

        return markov_sample_step
    spec = importlib.util.spec_from_file_location(source.stem, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.markov_sample_step


def measure(functions, warmup, iterations, repeats):
    graphs = {}
    outputs = {}
    samples = {name: {"graph_us": [], "eager_us": []} for name in functions}
    for name, fn in functions.items():
        for _ in range(warmup):
            outputs[name] = fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs[name] = fn()
        graphs[name] = graph
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            graphs[name].replay()
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name]["graph_us"].append(
                start.elapsed_time(end) * 1000 / iterations
            )
            before = time.perf_counter()
            for _ in range(iterations):
                outputs[name] = functions[name]()
            torch.npu.synchronize()
            samples[name]["eager_us"].append(
                (time.perf_counter() - before) * 1e6 / iterations
            )
    for name, sample in samples.items():
        sample["graph_median_us"] = statistics.median(sample["graph_us"])
        sample["eager_median_us"] = statistics.median(sample["eager_us"])
        graphs[name].reset()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    sample = load_sampler(args.source)
    baseline = load_sampler(args.baseline_source) if args.baseline_source else None
    dtype = getattr(torch, args.dtype)
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(),
                "torch": torch.__version__,
                "source_sha256": (
                    hashlib.sha256(args.source.read_bytes()).hexdigest()
                    if args.source
                    else None
                ),
            }
        ),
        flush=True,
    )
    for batch in args.batch:
        torch.manual_seed(42)
        base = torch.randn(batch, args.vocab, dtype=dtype, device="npu")
        bias = torch.randn_like(base)
        temperatures = torch.full((batch,), 0.7, device="npu")
        noise = torch.empty(batch, args.vocab, device="npu").exponential_()
        logits = torch.empty_like(base)
        kwargs = dict(
            base_logits=base,
            bias=bias,
            temperatures=temperatures,
            key=12345,
            offset=torch.tensor([7], device="npu"),
            step=2,
            step_dtype=dtype,
            logits_out=logits,
        )

        def native():
            value = (base.float() + bias.float()).to(dtype).float()
            return (value / temperatures[:, None] - noise.log()).argmax(-1)

        def total():
            noise.exponential_()
            return sample(**kwargs, exp_noise=noise)

        value = (base.cpu().float() + bias.cpu().float()).to(dtype).float()
        expected = (value / temperatures.cpu()[:, None] - noise.cpu().log()).argmax(-1)
        actual = sample(**kwargs, exp_noise=noise)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(logits.cpu(), value.to(dtype), rtol=0, atol=0)
        functions = dict(
            fixed_philox=lambda: sample(**kwargs),
            precomputed_kernel=lambda: sample(**kwargs, exp_noise=noise),
            native_same_noise=native,
            noise_generation=lambda: noise.exponential_(),
            generation_plus_kernel=total,
        )
        if baseline is not None:
            functions["baseline_philox"] = lambda: baseline(**kwargs)
        print(
            json.dumps(
                {
                    "batch": batch,
                    "vocab": args.vocab,
                    "dtype": args.dtype,
                    "noise_bytes": noise.numel() * noise.element_size(),
                    "tokens_exact": True,
                    "timings": measure(
                        functions, args.warmup, args.iterations, args.repeats
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
