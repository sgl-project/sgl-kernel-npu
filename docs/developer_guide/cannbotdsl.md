# DeepSeek-V4.1 CANNBot-DSL operators

Seven repository-owned source entries live in `implementation/cannbot-dsl/`:
K, QLI/QSLI and their metadata under `indexer/`; MLA and its metadata under
`attention/`. `UPSTREAM.json` records upstream paths, hashes and local backports.
The CANN license ships beside these files. The K projection tail-allocation fix
is retained: backing rows round up to 16 while the logical view keeps T rows.

## Installation and execution

Use the repository-root build with a clean source checkout. The existing build
stages the Python implementation tree, including the DSL sources and license;
no additional CMake target or changes to shared build scripts are needed.
For a wheel without third-party providers:

```bash
python3 setup.py --arch-version 3510 --without-third-party -j8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pip install 'cannbot-dsl==0.8.0'
```

Select the providers required by the deployment when building a full SGLang
wheel. The separate compressor still needs `cann_ops_transformer`; the example
above does not supply it. Install the DSL runtime explicitly; this contribution
does not add a wheel extra or precompile the seven JIT operators.

The selected target is A5 / ARCH 3510, CANN 9.2.0, DSL 0.8.0 and matching
Linux torch/torch_npu. The upstream source documents CANN 9.2.1. The pinned DSL
API is not compatible with every later runtime; update sources and runtime
together. `auto` selects `cannbot_dsl`. Warm actual shapes before graph capture.

The four compute entries also register `torch_native`: independent CPU Torch
references with outputs copied back to the input device. They preserve in-place
cache/output semantics but are not performance or graph-capture backends.
Metadata is a device-specific schedule, not a unique mathematical tensor;
therefore its three entries do not register a fake native alias.

## Accuracy and performance on A5

The standard public-interface tests cover decode T=1, DSPARK static verification
T=6 and selected boundaries. Inputs are synthesized from model dimensions, not
captured production tensors. K uses compressor width 512, index width 128 and
RoPE width 64; QLI/QSLI use 32 heads, 4096 keys and TopK 512; MLA uses 64 heads
of width 512 with original/compressed sparse lengths 128/512.

```bash
python3 -m pytest \
  tests/python/operators/indexer/test_indexer_prologue_k.py \
  tests/python/operators/indexer/test_quant_lightning_indexer.py \
  tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py \
  tests/python/operators/indexer/test_quant_sparse_lightning_indexer.py \
  tests/python/operators/indexer/test_quant_sparse_lightning_indexer_metadata.py \
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py \
  tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py \
  --npu-device=0 --skip-acc-check=false \
  --prof-dir=results/dsv41 --csv-path=results/dsv41.csv -v -s -x -rs
```

Only representative shapes enable profiling by default. Add `--enable-perf=false`
for accuracy only. The existing test tools record device task time in microseconds;
this is operator timing, not serving TPOT or a measured speedup.

K and MLA compare independently allocated mutable outputs. QLI/QSLI check scores,
valid indices and the TopK boundary, allowing equal-score permutations. A small
K allocation guard retains the T=1/T=17 overflow regression. The metadata tests
change lengths in place and compare graph replay with fresh eager schedules;
paired compute tests check the resulting schedule's numerical semantics.

CPU reference checks do not establish NPU precision, graph compatibility or
performance. Run the above tests on A5 after changes and retain the device results.
