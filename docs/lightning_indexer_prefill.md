# Ratio-4 LightningIndexer prefill grouping

This explicit API reduces the number of CANN batches and duplicate page-table
rows for one causal prefill request. It does not replace LightningIndexer
arithmetic, skip query scoring, or share Top-K results. It does not change model
dispatch or the existing Triton indexer.

```python
from sgl_kernel_npu.indexer.lightning_indexer_prefill import lightning_indexer_prefill

indices = lightning_indexer_prefill(q, k, sequence_length=sequence_length)
```

Q is contiguous BF16 `[R,4,128]`; K is contiguous BF16 `[N//4,1,128]` on the same
NPU. Q must represent the last R positions of that single request of length N.
All four query-head weights are one. Output is int32 `[R,512]`, ordered indices
local to K, padded with -1. Arbitrary masks, learned head weights, other shapes,
decode and speculative execution are outside this API. Callers keep their
existing fallback for unsupported workloads. No input device values are read
back for eligibility or phase calculation.

For row i, the visible key count is `floor((N-R+i+1)/4)`. The phase
`(N-R+1)%4` aligns four-query groups so all real queries in each group have the
same visible K range. Queries still produce independent scores and selections.
Leading/trailing dummy queries are discarded. CANN requires at least one key
in its length metadata, so rows with no visible key are masked back to -1.
Zero rows or zero K bypass CANN. R<1024 or K<512 uses independent queries.
`group4=False` explicitly selects that comparison path.

## Fresh extracted-source validation

Ascend910_9362 (910C), CANN 9.0, torch/torch-npu 2.10. The included unittest
passes 36 random/zero-input cases against a separately implemented one-query
CANN reference, with three queued calls per case. Cases cover all four phases,
partial groups, empty K, small K, short fallback, threshold boundaries and long
prefill. Eight retained real-model captured inputs also match all stored
ordered indices after extraction. This validates the adapter against CANN;
it is not a new independent correctness proof of CANN's arithmetic.

Fresh benchmark: three warmups, 20 alternating-order NPU event samples.
Both paths include Python adaptation, padding, table construction and masking.
Inputs are random nonzero BF16 tensors. Short-input differences are measurement
noise because both paths dispatch the same implementation.

| Query rows / keys | Independent ms | Group4 ms | Ratio |
| --- | ---: | ---: | ---: |
| 128 / 1024 | 0.447 | 0.445 | 1.00× |
| 1024 / 512 | 0.608 | 0.574 | 1.06× |
| 16384 / 4096 | 11.831 | 4.732 | 2.50× |
| 16384 / 8192 | 18.751 | 7.834 | 2.39× |
| 8313 / 11102 | 13.831 | 6.072 | 2.28× |

Raw samples: [lightning_indexer_group4_910c.json](benchmarks/lightning_indexer_group4_910c.json).

```bash
python -m unittest discover -s tests/python/sgl_kernel_npu -p test_lightning_indexer_prefill.py -v
python benchmark/indexer/bench_lightning_indexer_prefill.py --device 0 --samples 20 --output /tmp/indexer.json
```

## Historical serving context

The earlier local Whittle-Next-26B-A3B W8A8/TP1 overlay compared the same
QSA/GDN candidate before and after group4. Fixed budget2048, C6, 12 requests,
80,479 newly computed prompt tokens (cached tokens excluded): prefill
8708.48/8753.73 -> 9474.14/9522.00 new tokens/s (about +8.8%). Independent
12x256 output workload: 114.00 -> 117.47 tokens/s. Candidate max output TTFT
4.620s, max request-average TPOT 43.060ms; all three runs passed 12/12 SLOs.
100 model Indexer calls compared about 454 million ordered indices bitwise,
with 90 calls using group4. Fixed 384-token logprobs/Top5 matched the prior
candidate exactly.

These are sequential historical local-overlay measurements, not a fresh
end-to-end benchmark of this upstream PR. They span different service starts
and dates and have no confidence interval. The PR adds an API, not serving
integration. Original-model task quality and full Flash-Next/distributed TP
remain unaccepted. The adapted Whittle fixture's inert indexer weights do not
substitute for the nonzero random input tests above.
