# Original A5 LIM source bundle

`original-lim-pr18383-source.tar.xz` contains the CANN build tree needed for the
original LIM operator only. Its upstream code files are copied byte-for-byte
from `vllm-project/vllm-ascend` revision
`7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e` (PR 18383, before migration into the
current SGL repository). The archive also includes the upstream `LICENSE`.
SHA256: `5bfe5edd5c7b0a9f76001b9aaf93c86ffb7ebe4f9c790d675eb5e8fe0a335fff`.

The package contains `csrc/build.sh`, the CANN CMake/helpers/common files, and
`csrc/attention/fused_quant_lightning_indexer_manage`. Other CANN operators and
the vLLM Python tree are omitted. `prepare_original.py` records a SHA256 for
every copied upstream file in `SOURCE_MANIFEST.json`. It adds two empty CMake
category shims for `ffn` and `posembedding`, because the pinned root CMake calls
`add_subdirectory` for those missing optional directories unconditionally.
The LIM operator source itself is not patched. The LIM kernel uses the original
SIMT path and its original host tiling/OPP registration; this bundle does not
replace that path with an SGL direct launch. The LIM source has no Catlass
include or operator CMake dependency, so the Catlass submodule is not included.
Only `--pkg`/operator-package construction is in scope; static-library and UT
build paths that initialize repository submodules are not used.

## Offline build dependencies

The source CMake requests these fixed dependencies. They are included so a
prepared build does not need network access:

| Dependency | Source and version | SHA256 | Packaging |
| --- | --- | --- | --- |
| nlohmann/json headers | [v3.11.3](https://github.com/nlohmann/json/tree/v3.11.3/include) | `5580533576f6cde8f2c36843f3b27fc552d263bdcc00c597a36b1919a4a78df8` | The upstream `include/` tree is repacked as the `include.zip` expected by CANN CMake. Its `LICENSE.MIT` text is carried separately with trailing whitespace normalized. |
| Abseil | [20230802.1](https://github.com/abseil/abseil-cpp/tree/20230802.1) | `987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed` | Unmodified upstream source tarball. |
| Protocol Buffers | [v25.1](https://github.com/protocolbuffers/protobuf/tree/v25.1) | `9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42` | Unmodified upstream source tarball. |
| makeself | [release-2.5.0](https://github.com/megastep/makeself/tree/release-2.5.0) | `705d0376db9109a8ef1d4f3876c9997ee6bed454a23619e1dbc03d25033e46ea` | Installed into the source tree so CMake skips its download step. |

The CANN makeself URL is for a patched `release-2.5.0-patch1.0` archive and is
not anonymously retrievable from this host (GitCode returned HTTP 401). The
bundle therefore uses upstream makeself 2.5.0 for package creation. The pinned
CMake packaging commands use `--header`, `--help-header`, `--tar-format posix`,
`--gzip`, `--complevel 4`, `--nomd5`, and `--sha256`; all are implemented in the
included upstream `makeself.sh`. This is a package-wrapper dependency
substitution only, not an LIM kernel/source substitution. If the exact CANN
patched makeself build is required for release parity, replace this one
packaging dependency with its authenticated archive before making a result
claim.

The standard makeself runtime requires `--` before embedded installer options:
`bash package.run -- --install-path=/absolute/private/path`. Without that
separator, it rejects `--install-path` before the embedded CANN script starts.
`build_original.sh` and `recover_private.sh` use the separator. A CPU regression
creates and runs a real archive with this vendored makeself release.

The CANN CMake dependency patches for Abseil and protobuf are preserved without
modification in the source bundle. CPU-side patch syntax/application was
checked against the corresponding upstream source tags. A5 compilation,
installer execution, OPP registration, numerical correctness, graph capture,
and performance still require validation on the target A5/CANN installation.

PR 18490 is not used by this baseline or included in the bundle. It is only a
separate source-history reference for the SGL migration. The manual LIM
benchmark must build and invoke the original ACLNN/OPP operator from this PR
18383 source package as its original backend.
