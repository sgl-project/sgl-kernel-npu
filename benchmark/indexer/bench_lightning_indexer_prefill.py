import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def measure(functions, samples):
    for fn in functions:
        for _ in range(3):
            fn()
    torch.npu.synchronize()
    times = [[] for _ in functions]
    for i in range(samples):
        for j in (
            range(len(functions)) if i % 2 == 0 else reversed(range(len(functions)))
        ):
            a = torch.npu.Event(enable_timing=True)
            b = torch.npu.Event(enable_timing=True)
            a.record()
            functions[j]()
            b.record()
            b.synchronize()
            times[j].append(a.elapsed_time(b))
    return {"median_ms": [statistics.median(t) for t in times], "samples_ms": times}


from sgl_kernel_npu.indexer.lightning_indexer_prefill import lightning_indexer_prefill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(4201)
    records = []
    for rows, n in (
        (128, 4096),
        (1024, 2048),
        (16384, 16384),
        (16384, 32768),
        (8313, 44408),
    ):
        q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(n // 4, 1, 128, device="npu", dtype=torch.bfloat16)
        funcs = [
            lambda grouped=grouped: lightning_indexer_prefill(
                q, k, sequence_length=n, group4=grouped
            )
            for grouped in (False, True)
        ]
        assert torch.equal(funcs[0](), funcs[1]())
        records.append(
            dict(
                rows=rows,
                sequence_length=n,
                keys=n // 4,
                ordered_indices_equal=True,
                **measure(funcs, args.samples)
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
