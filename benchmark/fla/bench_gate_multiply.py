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


from sgl_kernel_npu.fla.gate_multiply import gate_multiply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(100326)
    records = []
    for tokens in (1024, 2048, 3072, 4096, 4621, 16384):
        y = torch.randn(tokens, 4096, device="npu", dtype=torch.bfloat16)
        z = torch.randn_like(y)
        # Include the unchanged sigmoid cost in both paths.
        funcs = [
            lambda: (y.float() * torch.sigmoid(z.float())).to(y.dtype),
            lambda: gate_multiply(y, torch.sigmoid(z.float())),
        ]
        assert torch.equal(funcs[0](), funcs[1]())
        records.append(
            dict(
                tokens=tokens,
                elements=y.numel(),
                bitwise_equal=True,
                **measure(funcs, args.samples)
            )
        )
    args.output.write_text(
        json.dumps(
            dict(
                device=torch.npu.get_device_name(args.device),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                order=["torch_multiply_cast", "fused_helper"],
                includes_sigmoid=True,
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
