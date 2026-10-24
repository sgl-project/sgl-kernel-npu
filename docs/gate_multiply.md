# BF16 gate multiply / cast fusion

```python
from sgl_kernel_npu.fla.gate_multiply import gate_multiply

# Keep the existing normalization and sigmoid implementations.
gate = torch.sigmoid(z.float())
out = gate_multiply(normalized_bf16, gate)
```

This inference-only helper computes `(y.float() * gate).to(torch.bfloat16)`.
It accepts matching BF16 y and FP32 gate on the same NPU. Normalization has
already rounded y to BF16; that rounding is preserved. Sigmoid is not fused
or approximated. This is not a replacement for the existing swish-gated
normalization kernel and does not modify model dispatch.

The fused Triton NPU kernel loads y as BF16, promotes to FP32, multiplies by
the stored FP32 gate and stores BF16. It removes the full-size promoted-y and
FP32 product intermediates from this helper. At least 16,777,216 contiguous
elements are required; smaller or strided inputs retain the PyTorch expression.
The threshold comes from 910C measurements, not all NPU platforms. The cached
vector-core count is keyed by device. Runtime element count does not specialize
the kernel. Warm the shape/kernel before graph capture.

## Fresh extracted-source validation

Ascend910_9362 (910C), CANN 9.0, torch/torch-npu 2.10. Four included unittest
methods pass: empty/small inputs and both sides of the threshold, three queued
temporary-input calls for each size, large strided fallback, sigmoid saturation,
NaN/Inf/signed-zero comparisons, invalid inputs, and five warmed graph replays
with changed inputs. Finite outputs are bitwise equal to the PyTorch expression;
NaN locations are compared separately.

Fresh benchmark: three warmups, 20 alternating-order NPU event samples.
Both paths include the same `torch.sigmoid(z.float())` cost, but exclude
normalization. Cases below 4096 tokens use the PyTorch fallback. Differences there reflect
wrapper/measurement effects, not fused-kernel speedups. The extracted PR uses
a more conservative threshold than the historical overlay (8,388,608 elements).

| Tokens (4096 elements each) | PyTorch ms | Helper ms | Ratio |
| --- | ---: | ---: | ---: |
| 1024 | 0.110 | 0.120 | 0.92× |
| 2048 | 0.169 | 0.134 | 1.26× |
| 3072 | 0.312 | 0.312 | 1.00× |
| 4096 | 0.499 | 0.281 | 1.77× |
| 4621 | 0.568 | 0.350 | 1.62× |
| 16384 | 2.150 | 1.253 | 1.72× |

Raw samples: [gate_multiply_910c.json](benchmarks/gate_multiply_910c.json).

```bash
python -m unittest discover -s tests/python/sgl_kernel_npu -p test_gate_multiply.py -v
python benchmark/fla/bench_gate_multiply.py --device 0 --samples 20 --output /tmp/gate.json
```

## Historical serving context

Earlier local Whittle-Next-26B-A3B W8A8/TP1 integration retained the existing
normalization and torch sigmoid and replaced only the final multiply/cast.
270 real calls / 10,895,892,480 output elements were bitwise equal to the prior
group4 candidate, as were fixed 384-token logprobs and complete Top5 lists.
Fixed budget2048/C6/12 requests/80,479 new prompt tokens, excluding cache hits:
9636.87/9692.60 new tokens/s; independent 12x256 output: 118.78 tokens/s.
Max output TTFT 4.546s, max request-average TPOT 42.626ms; three runs passed
12/12 SLOs. Same-day group4 control was 9495.86 prefill / 117.37 output tokens/s,
so sequential gains were about +1.49% / +1.20%, without a confidence interval.

Separate diagnostic events over six C1 requests / 42,441 new tokens, excluding
36,155 warmup tokens, showed norm+gate 200.55 -> 125.20ms (-37.57%), GDN full
1488.33 -> 1412.69ms (-5.08%), model events 4548.80 -> 4461.34ms (-1.92%).
Nested event intervals are diagnostic, not clean serving throughput.

These are historical local-overlay results, not a fresh end-to-end benchmark
of this upstream API. Serving integration requires a separate caller change.
Original-model business quality remains unaccepted; equivalence here is to
the previously validated candidate.
