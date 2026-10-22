"""Event timing for shared, independent, and mixed selected QSA blocks."""

import argparse
import json
import statistics

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.attention.qsa_prefill import qsa_prefill


def main(args):
    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    torch.manual_seed(9302026)
    records = []
    for rows in args.rows:
        length = rows + 8192
        q = torch.randn(rows, 16, 256, device="npu", dtype=torch.bfloat16)
        k = torch.randn(length, 2, 256, device="npu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        table = torch.arange(length, device="npu", dtype=torch.int32)[None]
        req = torch.tensor([0], device="npu", dtype=torch.int64)
        for mode in ("shared", "different", "mixed"):
            blocks = torch.arange(512, dtype=torch.int32).repeat(rows, 1)
            if mode == "different":
                blocks += torch.arange(rows, dtype=torch.int32)[:, None] % 500
            elif mode == "mixed":
                blocks[1::4] += 3
            blocks = blocks.npu()
            call = (q, k, v, blocks, table, req, length, length - rows)
            for _ in range(3):
                qsa_prefill(*call)
            torch.npu.synchronize()
            times = []
            for _ in range(args.iterations):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                    enable_timing=True
                )
                start.record()
                qsa_prefill(*call)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            row = dict(
                rows=rows,
                mode=mode,
                median_ms=statistics.median(times),
                samples_ms=times,
            )
            records.append(row)
            print(json.dumps(row), flush=True)
        del q, k, v, table, req, blocks, call
        torch.npu.empty_cache()
    if args.output:
        with open(args.output, "w") as out:
            json.dump(records, out, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rows", type=int, nargs="+", default=[128, 4621, 16384])
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output")
    main(p.parse_args())
