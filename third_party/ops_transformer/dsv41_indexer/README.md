# A3 CANN 9.2 INT8 indexer integration

The existing public `quant_lightning_indexer` interface selects three modes:

| Mode | Existing ops-transformer experimental operator |
| --- | --- |
| 1: source | QuantLightningIndexerV2 with 2048 candidate blocks of size 8 |
| 2: consumer | QuantSparseLightningIndexer using those candidate block IDs |
| 3: ordinary | QuantLightningIndexerV2 with candidate output disabled |

These operators and their shared QuantLightningIndexerV2Metadata are built from
ops-transformer `e7da18105f08b06e01ef0fff78f439098d483746`, already pinned by the
mainline CANN 9.2 profile. No computation kernel, ACLNN signature, or C++ binding
is copied or changed. The wheel stages two small FakeTensor dtype/device fixes.
The A3 build corrects the two indexers' host workspace formulas: the current
arch22 kernels store head-reduced 4-row scores, while upstream still reserves
256 rows. The FP16 weight workspace is explicitly retained, sized for 64 heads;
the existing decode and candidate reduction buffers are retained unchanged.
CANN 9.2's built-in released V2 API does not expose candidate input/output;
the experimental vendor must be built and installed. A2/A5 and CANN 9.1 retain
their existing selections. Legacy callers still receive a pair; explicit
candidate_mode=1/2/3 returns a triple, without adding an SGLang operator name.

Q/K are INT8 and their per-row scales and head weights are FP16. SGLang uses
`torch_npu.npu_dynamic_quant` for Q/K. Attention KV remains BF16.

```sh
python3 setup.py --arch-version 2201 -j 16
python -m pip install --force-reinstall --no-deps output/sgl_kernel_npu*.whl
python -m pytest -q tests/cpu/test_quant_lightning_indexer_modes.py
python -m pytest -q tests/python/operators/indexer/test_quant_lightning_indexer.py tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py --enable-perf=false
```

Use the SGLang companion branch `codex/dsv41-a3-cann92-int8-indexer`.
Real A3 replay, end-to-end GSM8K accuracy, and performance qualification are
required after replacing the prior vendored implementation with this route.
