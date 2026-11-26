# LightningIndexer Prefill Benchmark

Benchmark results comparing four-query batching (`group4=True`) against
independent-query dispatch (`group4=False`) for
`sgl_kernel_npu.operators.indexer.lightning_indexer_prefill`.

## Reproduction

```bash
python3 benchmark/indexer/bench_lightning_indexer_prefill.py \
  --device 0 --samples 20 --output /tmp/lightning_indexer_prefill_bench.json
```

## Methodology

Three warmups followed by 8 to 20 alternating-order NPU event samples per
shape. Both columns include the Python adaptation, padding, page-table
construction, and masking on random nonzero BF16 inputs.

## Results

### Ascend910_9362 (A3 / 910C, CANN 9.0)

| Query rows / keys | `group4=False` | `group4=True` | Ratio |
| --- | ---: | ---: | ---: |
| 128 / 1024 | 0.447 ms | 0.445 ms | 1.00x |
| 1024 / 512 | 0.608 ms | 0.574 ms | 1.06x |
| 16384 / 4096 | 11.831 ms | 4.732 ms | 2.50x |
| 16384 / 8192 | 18.751 ms | 7.834 ms | 2.39x |
| 8313 / 11102 | 13.831 ms | 6.072 ms | 2.28x |

### Ascend910B3 (A2, CANN 9.1.0, 5-AI-core virtualized slice)

| Query rows / keys | `group4=False` | `group4=True` | Ratio |
| --- | ---: | ---: | ---: |
| 128 / 1024 | 0.721 ms | 0.726 ms | 1.00x |
| 1024 / 512 | 1.319 ms | 1.022 ms | 1.29x |
| 16384 / 4096 | 31.201 ms | 13.184 ms | 2.37x |
| 16384 / 8192 | 65.601 ms | 27.390 ms | 2.39x |
| 8313 / 11102 | 49.188 ms | 21.025 ms | 2.34x |

Shapes below the grouping thresholds (`rows < 1024` or `keys < 512`) dispatch
the same independent-query work in both columns, so their differences are
measurement noise.
