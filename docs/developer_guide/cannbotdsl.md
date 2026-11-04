# DeepSeek-V4.1 CANNBot-DSL operators

The seven public entries use repository-owned source, pinned to
`cann/cannbot-dsl@71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`. The source and its
CANN Open Software License v2.0 are under
`python/sgl_kernel_npu/sgl_kernel_npu/implementation/cannbot-dsl/`.
`UPSTREAM.json` records the original paths and SHA-256 hashes. The seven upstream
precision suites retain independent golden calculations; their imports now call
the public SGL APIs. They are the correctness reference instead of a second
production backend. `auto` and `cannbot_dsl` select the same implementation.

**Target: CANN 9.2.0, cannbot-dsl 0.8.0, A5 / ARCH 3510. Device compilation,
accuracy, graph replay and performance have not been validated for this
combination.** Upstream master documents CANN 9.2.1; the selected 9.2.0 target
must pass the tests below before a compatibility claim. Other architectures are
rejected by the generation-aware registry before importing the DSL backend.

## Public API

```python
from sgl_kernel_npu.operators.indexer import (
    indexer_prologue_k,
    quant_lightning_indexer,
    quant_lightning_indexer_metadata,
    quant_sparse_lightning_indexer,
    quant_sparse_lightning_indexer_metadata,
)
from sgl_kernel_npu.operators.attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)
```

These replace the seven vendor `ops.*` imports in SGLang's `_ds41_ops`.
`compressor` and other external operators are outside this migration and retain
their existing dependencies. Signatures match the pinned DSL source, with an
additional keyword-only `impl="auto"`. MLA writes the caller's `out` and `lse`;
K prologue mutates its cache arguments. No implicit tensor conversion is added.
Metadata and compute must use the same layouts, lengths and static parameters.

## Source execution and optional Native bundle

As with the Triton implementations, normal wheels include the Python source.
Install the optional backend into the target Linux environment:

```bash
python3 -m pip install 'cannbot-dsl==0.8.0'
```

The runtime checks the exact DSL version when a DSL entry is first selected.
Without a Native bundle, upstream JIT compilation is used; compilation must be
warmed before graph capture. Python 3.10-3.12 and a matching torch/torch_npu pair
are required. The legacy wheel also exposes the `cannbot_dsl` dependency extra.

To compile AICore providers and all three AICPU libraries while building the
sgl-kernel-npu wheel, run from this repository's root on the target build system:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install 'cannbot-dsl==0.8.0'

# Inspect the DSL plan only. This does not compile or query an NPU.
python3 setup.py --arch-version 3510 --without-third-party \
  --with-cannbot-dsl --plan --cann-version 9.2.0

# Compile the new-build native operators and the seven DSL entries together.
python3 setup.py --arch-version 3510 --without-third-party --with-cannbot-dsl -j8
```

`--without-third-party` here isolates the repository-owned migration. Omit it,
or select the required `--third-party` profiles, when composing a complete
deployment wheel. These commands do not claim to package all legacy native ops.
The DSL option is supported only by the new wheel build, not the `.run` or
legacy `build.sh` path. The build checks the actual SDK's exact version; the
`--cann-version` override remains plan-only.

Native compilation uses the upstream export profiles. The indexers export many
static variants by default; `-j` does not control the DSL compiler. To restrict
coverage, set `CANNBOTDSL_DS41_PROFILES` to a JSON file using the pinned
`export_indexer_prologue_k`, `export_qli` and `export_qsli` fields. Its content
hash is recorded. A restricted profile must cover the actual model's static
parameters; a successfully built wheel is not proof of that coverage.

The bundle includes `_native/`, `_aicpu/` and a build manifest beside the seven
source files. AICPU libraries use separate build directories to avoid the two
`metadata_kernel` symbols colliding. Packaging checks resource hashes and the
compiled source hashes. DSL-enabled wheels depend on `cannbot-dsl==0.8.0` and
have a `cb_<fingerprint>` build tag. `--make-clean` removes DSL intermediates.

## Validation

Offline dispatch/resource tests do not execute any kernel:

```bash
python3 -m pytest tests/unit/test_cannbotdsl_packaging.py tests/unit/test_cannbotdsl_runtime.py -q
```

On A5, install the built wheel and run the public API tests. Tests retain the
independent upstream reference calculations, including MXFP4 ties, candidate selection,
cache strides and partial pages. No performance result is implied by these
accuracy tests.

```bash
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
export CANNBOTDSL_RUN_SMLA_TESTS=1
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl -q
```

For Native bundle validation, repeat the required production cases with:

```bash
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT="$PWD/native-hits.json"
python3 -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl -q
```

The full upstream edge-case matrix may need extra static profiles. A Native
miss is a profile-coverage failure, not evidence that the numerical algorithm
failed. Require-mode also rejects missing bundled AICPU resources before the
metadata compiler can silently fall back to JIT.

`test_metadata_graph.py` checks all three scheduling operators after lengths
change between replays. Full compute graph capture and model-level PD/cache
integration still require separate A5 validation. Keep these pending until
actual device results are available.
