# DeepSeek-V4.1 CANNBot-DSL operators

The seven attention/indexer entries ship as repository-owned Python source,
like the Triton implementations. `implementation/cannbot-dsl/UPSTREAM.json`
records the source commit, original paths and hashes; the CANN license is
packaged beside it. `auto` and `cannbot_dsl` select the same implementation.

The selected validation target is **A5 / ARCH 3510, CANN 9.2.0 and
cannbot-dsl 0.8.0**. Upstream documents CANN 9.2.1; the selected combination
still needs device compilation, numerical and graph-replay validation.

## Build and use

Build the ordinary kernel wheel using the repository-root build command:

```bash
python3 setup.py --arch-version 3510 --without-third-party -j8
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pip install 'cannbot-dsl==0.8.0'
```

`--without-third-party` isolates the migrated sources. Select the required
third-party profiles when composing a complete deployment wheel. This command
does not imply that all operators needed by SGLang are included.

DSL functions are imported and JIT-compiled on use. Warm the actual shapes
before graph capture. No precompiled DSL bundle is produced by this build.
The optional `cannbot_dsl` wheel extra installs the DSL dependency; unrelated
operators do not require it. Use a matching Linux/Python and torch/torch_npu
stack (the A5 validation environment uses Python 3.12 and torch/torch_npu 2.13.0).

Import the public entries from `sgl_kernel_npu.operators.indexer` (K prologue,
QLI/QSLI and metadata) and `sgl_kernel_npu.operators.attention` (MLA and metadata).
Their signatures match the pinned sources with an extra `impl="auto"` keyword.
MLA writes caller-owned `out`/`lse`; K prologue updates its cache arguments.
Inputs must share an A5 device. The separate AscendC compressor still requires
`cann_ops_transformer` in SGLang.

## Validate on A5

```bash
export CANNBOTDSL_RUN_SMLA_TESTS=1
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl -q
```

The suites retain independent numerical references. `test_metadata_graph.py`
checks all three metadata operators after lengths change between replays.
`scripts/check_cannbotdsl_ci.py` checks the pinned environment, installed
artifacts and all required core JUnit cases; skipped cases are not accepted.
Full compute graph capture, model/PD integration and performance require
separate A5 validation. Offline and fixture-wheel tests do not establish
hardware compatibility or speedup.
