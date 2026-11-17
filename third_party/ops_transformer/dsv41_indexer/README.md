# A3 CANN 9.2 INT8 indexer source overlay

This directory carries the QLI V2 compute and metadata implementation from
vLLM-Ascend `9e3cc3dc223e91b7892edc787f1e15e47d007ff0` (see UPSTREAM.json).
The original CANN license headers are preserved. It replaces the two existing
operator source directories in the pinned ops-transformer build checkout only
when building the CANN 9.2 profile for Ascend910_93. A2, A5 and CANN 9.1 use
their original recipes. Pinned CMake rules are retained except for the QLI
host rule, which must match the overlay's tiling implementation (no release
checker sources).
The overlay hashes are included in the resolved build/cache fingerprint.

No new public operator name is registered. `quant_lightning_indexer` gains
candidate_mode 1/2/3 and returns three tensors when the mode is explicitly
supplied; legacy callers still get two tensors. The matching metadata remains
`quant_lightning_indexer_metadata`. Both the AICore compute implementation and
the AICPU metadata library must be rebuilt and installed together.

The A3 binding restores the original CANN operator names, restricts the native
adapter to INT8, and uses the existing PyTorch dispatcher name for graph capture.
Q/K are INT8; weights and per-row Q/K scales are FP16. Q quantization in SGLang
uses torch_npu.npu_dynamic_quant as requested; it is not bitwise equivalent to
vLLM's FP16-scale-first query quantization kernel.

## Validation

Host checks, from sgl-kernel-npu:

```sh
python -m pytest -q tests/cpu/test_quant_lightning_indexer_modes.py
```

On an A3 Linux host with CANN 9.2 initialized, rebuild and install the wheel:

```sh
python3 setup.py --arch-version 2201 -j 16
python -m pip install --force-reinstall --no-deps output/sgl_kernel_npu*.whl
python -m pytest -q tests/python/operators/indexer/test_quant_lightning_indexer.py tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py
```

Use the SGLang companion branch `codex/dsv41-a3-cann92-int8-indexer` with
PYTHONPATH pointing to its python directory. Its A3 V4.1 path selects INT8
without a new launch flag; attention KV and SWA remain BF16. Run its A3 backend,
cache and DSpark tests before server accuracy/performance qualification.

The new device tests cover C1/C2, source/consumer/ordinary modes, and NPUGraph
replay with changed K scales and KV lengths through source and consumer. CPU reference/schema tests do not establish CANN
build success, ACL Graph correctness, end-to-end accuracy or performance.
