"""Compare the grouped and independent LightningIndexer prefill geometries."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

# Prefill chunks spanning the grouping threshold and production lengths.
SHAPES = (
    (128, 4096),
    (1024, 2048),
    (16384, 16384),
    (16384, 32768),
    (8313, 44408),
)


def measure(functions, samples):
    """Median and raw per-call milliseconds, sampled in alternating order."""
    for fn in functions:
        for _ in range(3):
            fn()
    torch.npu.synchronize()
    times = [[] for _ in functions]
    for i in range(samples):
        order = range(len(functions))
        for j in order if i % 2 == 0 else reversed(order):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            functions[j]()
            end.record()
            end.synchronize()
            times[j].append(start.elapsed_time(end))
    return {"median_ms": [statistics.median(t) for t in times], "samples_ms": times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(4201)
    records = []
    for rows, sequence_length in SHAPES:
        q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(
            sequence_length // 4, 1, 128, device="npu", dtype=torch.bfloat16
        )
        functions = [
            lambda grouped=grouped: lightning_indexer_prefill(
                q, k, sequence_length=sequence_length, group4=grouped
            )
            for grouped in (False, True)
        ]
        # The two geometries must select the same keys in the same order.
        assert torch.equal(functions[0](), functions[1]())
        records.append(
            dict(
                rows=rows,
                sequence_length=sequence_length,
                keys=sequence_length // 4,
                ordered_indices_equal=True,
                **measure(functions, args.samples)
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                device=torch.npu.get_device_name(args.device),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                order=["independent", "group4"],
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "samples_ms"} for r in records],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
