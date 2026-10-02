# torch.ops.npu.compressor

## Product Support Status

| Product | Arch | Supported |
| --- | --- | :---: |
| Atlas A3 Inference Product Series | `arch22` | √ |
| Atlas A5 Inference Product Series | `arch35` | √ |

## Function Description

`Compressor` implements the DeepSeek-V4 KV compression primitive. Every `cmp_ratio`
input tokens of a request are compressed into a single KV row. For each row the
kernel computes a windowed score, applies a stable softmax over the compression
window, produces a weighted sum of the window KV, then applies RMSNorm and RoPE
to emit the compressed KV row.

The operator also maintains a per-request rolling state (`state_cache`) used to
carry the score/KV context across calls. With `cache_mode=2` the state is a
per-request ring buffer addressed by a request bank id (`state_block_table`),
which lets long decode runs and speculative (MTP) verify reuse one fixed bank
per request without re-deriving state locations every step.

## Function Prototype

```
torch.ops.npu.compressor(
    x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos, *,
    rope_head_dim, cmp_ratio, state_block_table, cu_seqlens, seqused, start_pos,
    coff, norm_eps, rotary_mode, cache_mode, state_cache_stride_dim0=0) -> Tensor
```

## Parameter Description

- **x** (`Tensor`): required. Token hidden states. TH layout, shape `[T, H]`.
  `T` is the total number of tokens of the batch, `H` the hidden size.
  Supported dtypes: `bfloat16`, `float16`.
- **wkv** / **wgate** (`Tensor`): required. Fused compressor weights, same dtype
  as `x`.
- **state_cache** (`Tensor`): required, in/out. FP32 rolling state,
  shape `[block_num, ring_size, 2 * coff * head_dim]`.
- **ape** / **norm_weight** (`Tensor`): required, FP32. Positional score bias and
  RMSNorm weight.
- **rope_sin** / **rope_cos** (`Tensor`): required. RoPE tables for the compressed
  rows.
- **rope_head_dim** (`int`): required. RoPE head dimension (64 for DSV4).
- **cmp_ratio** (`int`): required. Compression ratio, `4` or `128`.
- **state_block_table** (`Tensor`): required. INT32 `[B]` request bank ids
  (ring mode); each active request owns one bank.
- **cu_seqlens** (`Tensor`): required for TH. INT32 `[B + 1]` prefix sums of the
  per-batch input capacity.
- **seqused** (`Tensor`): required. INT32 `[B]` number of valid tokens this call
  (`1` for decode, draft count for MTP verify, `0` for padding/idle slots).
- **start_pos** (`Tensor`): required. INT32 `[B]` absolute position of the first
  token of this call (committed sequence length for verify).
- **coff** (`int`): required. `1` or `2` (compression overlap).
- **norm_eps** (`double`): required. RMSNorm epsilon.
- **rotary_mode** (`int`): required. `2` (interleave) for DSV4.
- **cache_mode** (`int`): required. `2` = ring (used by sglang): `CYCLE` on
  arch35, `EXPLICIT` on arch22. `1` (paged/CONTINUOUS) exists in the kernel but
  is not used by sglang.
- **state_cache_stride_dim0** (`int`): optional, default `0`. When `0`, the host
  derives it as `state_cache.size(1) * state_cache.size(2)`.

## Variants

| Variant | head_dim | coff | cmp_ratio | Usage |
| --- | ---: | ---: | ---: | --- |
| C4A | 512 | 2 | 4 | Main C4 compressor |
| C4Li | 128 | 2 | 4 | Indexer compressor |
| C128A | 512 | 1 | 128 | Main C128 compressor |

## Build and Architecture Selection

This operator is one of the **dual-arch** operators in `sgl-kernel-npu`: a single
source tree is compiled for both the A2/A3 family (`arch22`) and the A5 family
(`arch35`). The architecture is chosen at configure time from `SOC_VERSION` in
`csrc/CMakeLists.txt`:

```cmake
# csrc/CMakeLists.txt
if(_sgl_soc_lower MATCHES "^ascend950")   # Ascend950* -> A5
    set(SGL_KERNEL_ARCH arch35)
else()                                    # Ascend910B* / A3 -> A2/A3
    set(SGL_KERNEL_ARCH arch22)
endif()

# Expose the arch family to host compilation (SGL_KERNEL_ARCH_22 / _35).
if(SGL_KERNEL_ARCH STREQUAL "arch35")
    add_compile_definitions(SGL_KERNEL_ARCH_35)
else()
    add_compile_definitions(SGL_KERNEL_ARCH_22)
endif()
```

`SGL_KERNEL_ARCH_35` (and its `_22` counterpart) is therefore **operator-agnostic
build metadata** that this operator uses to pick its own arch implementation:

- **Host tiling** (`op_host/compressor.cpp`):
  `#ifdef SGL_KERNEL_ARCH_35` includes `arch35/compressor_tiling.h`; otherwise
  `arch22/compressor_tiling.h`. CMake appends the matching
  `op_host/${COMPRESSOR_ARCH_DIR}/compressor_tiling.cpp` (where
  `COMPRESSOR_ARCH_DIR == SGL_KERNEL_ARCH`) to the build, so only one host tiling
  implementation is compiled per SoC.
- **Kernel** (`op_kernel/compressor.cpp`): the include path selects
  `op_kernel/arch22` or `op_kernel/arch35`; `__has_include("compressor_kernel_perf.h")`
  then resolves the arch22 body and otherwise the arch35 body. On A5 the compiler
  defines `__CCE_AICORE__ == 310`, which gates the arch35 vector micro-kernels
  (`op_kernel/arch35/vf/*`) used by the arch35 vector path.

Note the distinction from `SGL_KERNEL_ENABLE_A3_ONLY_OPS` / `_A5_ONLY_OPS`: those
are family-only toggles (an operator exists on one family only), while
`SGL_KERNEL_ARCH` is the arch selector for operators that exist on **both**
families — which is why compressor uses the latter.

The kernel entry dispatches on `tilingKey`, whose bits encode
`layout | dtype | coff | rotaryMode | cacheMode | templateId`.

`templateId` supports `NORMAL`, `EMPTY_X` (empty input, returns immediately) and
`FULL_LOAD` (a BSH + small-batch high-performance template). `FULL_LOAD` is only
selected by the host for `ASCEND950 + BSH + seqSize <= 4 + tokenSize <= 256`;
sglang uses TH + `cache_mode=2` and therefore always runs `NORMAL`. BSH dispatch
keys are not compiled in this build.

## arch22 vs arch35 Differences

Both arch implementations share the public op signature and the `tilingKey` bit
layout, but they differ in template selection, state-mode naming and vector
kernels. sglang uses TH + `cache_mode=2` + `rotary_mode=2` on both.

| Aspect | arch22 (A2/A3) | arch35 (A5) |
| --- | --- | --- |
| `cache_mode=2` name | `EXPLICIT` (ring) | `CYCLE` (ring) |
| Default `templateId` | `PERF` (2) | `NORMAL` (0) |
| High-perf template | `PERF` (`compressor_kernel_perf.h`), used for every non-empty call | `FULL_LOAD` (`compressor_kernel_full_load.h`), BSH + small batch only, unused by sglang |
| Kernel body | `compressor_kernel_perf.h` / `block_*_perf.h` | `compressor_kernel.h` / `block_*.h` |
| Vector softmax | `soft_max.h` (`ColumnSoftMax` / `ColumnMax`) | `vf/vf_softmax.h` (`SoftmaxDndBase*`) |
| RoPE / RMSNorm | `rope.h`, `rms_norm.h` (inline) | `vf/vf_rope.h`, `vf/vf_rms_norm.h` |
| Mul / Add | inline in `compressor_vector_comm.h` | `vf/vf_mul.h`, `vf/vf_add.h` |
| `readGen` handshake | not zeroed by the host (different sync) | host zeroes the AIV db release counters before launch |
| Empty input | `EMPTY_X` (templateId 1) | `EMPTY_X` (templateId 1) |

Input handling that differs:

- **`state_block_table`**: arch35 addresses one CYCLE ring bank per request
  (`req_pool_indices`); arch22 uses the EXPLICIT ring table ABI, which the sglang
  A3 backend builds explicitly.
- **`state_cache_stride_dim0`**: derived by the host when `0`; both arch22 and
  arch35 validate that `state_cache` is contiguous on its first axis.
- **`rotary_mode`**: sglang uses `2` on both; the host rejects other values (with
  TH) for the compiled key set.

## arch35-only Additions

Beyond the shared dual-arch code, arch35 carries several A5-only pieces:

**Host — AIV db release-counter (`readGen`) zeroing** (`op_host/compressor.cpp`,
inside `#ifdef SGL_KERNEL_ARCH_35`): before launching the kernel the host zeroes
the AIV double-buffer release counters so the cross-core generation handshake
starts from a known state:

```cpp
int64_t genFlagsSize = aivNum * dbWorkspaceRatio * sizeof(uint32_t);
int64_t libapiSize  = PlatformAscendCManager::GetInstance()->GetLibApiWorkSpaceSize();
int64_t gmSize      = (usedCoreNum + 1 + aivNum) * dbWorkspaceRatio * sizeof(uint32_t);
int64_t flagsOffset = workspaceSize - libapiSize - gmSize;   // kernel data tail, not - genFlagsSize
if (flagsOffset >= 0) {
    workspace.narrow(0, flagsOffset, genFlagsSize).view(at::kInt).zero_();
}
```

The kernel's `readGen` sits right after its data workspace (`InitWorkspace`
offsets start at 0), and the tiling workspace reserves the `(aicNum+1+aivNum)`
GM slab plus the libapi prefix at the end — hence `workspaceSize - libapiSize -
gmSize`, **not** `workspaceSize - libapiSize - genFlagsSize`. arch22 uses flags,
not GM, so this tail is only reserved/cleared on arch35.

**Why the dbIdx release side uses a GM generation counter instead of pure
flags (CYCLE mode):**

- **Need.** In CYCLE mode each db buffer (`dbIdx`, `dbWorkspaceRatio` of them) is
  reused across many rounds/generations. The kernel must be able to express
  "this dbIdx is at generation N and may now be overwritten".
- **Pure flags cannot express that.** (a) CrossCore flag ids are a limited
  resource, so dedicating channels per dbIdx/generation does not scale; (b) a
  single flag represents one event, not a generation — reusing the same flag
  across rounds allows same-flag reordering (a previous round's set being
  consumed by a later round's wait), so a consumer can be released early and
  overwrite a db buffer the AIV has not finished reading, producing wrong
  numbers.
- **GM generation counter fixes it.** One GM slot per `(AIV block, dbIdx)`
  (`GetBlockIdx() * dbWorkspaceRatio + dbIdx`) holds a monotonically increasing
  generation: the releaser stores `gen + 1`, and the consumer polls
  `while (readGenGm.GetValue(...) < gen)`. This has no channel limit and no
  same-flag ambiguity, so any number of rounds is ordered safely.
- **Readiness still uses a flag.** The mm1/Fixpipe "data has landed" signal
  keeps the hardware `CrossCore` flag (`C1_V1_FLAG`), whose pipe semantics
  guarantee the write is visible before consumers are released — a scalar GM
  write cannot provide that ordering guarantee.

This GM `readGen` handshake (replacing the pure-flag dbIdx release ordering) was
introduced on this branch while fixing the arch35 numerics (commits `00062f7`,
`1cee9dd`, `45e34c0`).

**Host tiling (arch35)**:

- `SetTemplateId`: on `ASCEND950`, selects `FULL_LOAD` only for
  `BSH + seqSize <= 4 + tokenSize <= 256` (sglang uses TH, so it stays `NORMAL`).
- `SetInnerSplitInfo`: the `FULL_LOAD` branch computes the m/d/k core split.
- `CheckFeature`: validates `state_cache` first-axis contiguity
  (`cacheStride == stateCacheStrideDim0`) on both arch22 and arch35.
- `cache_mode=2` is `CYCLE`: one ring bank per request, addressed by
  `state_block_table`.

**Kernel (arch35)**:

- Vector micro-kernels under `op_kernel/arch35/vf/`
  (`vf_softmax.h`, `vf_rope.h`, `vf_rms_norm.h`, `vf_mul.h`, `vf_add.h`),
  enabled when `__CCE_AICORE__ == 310`.
- `compressor_kernel.h` plus the `*_full_load.h` variants.
- The `readGen` generation handshake and CYCLE ring addressing.

## Operator-specific Internals

A few pieces are specific to this operator and worth knowing when reading the
code:

- **Custom host tiling framework (`ge_helper.h`)** — instead of the msopgen
  tiling path, the host builds a `TilingContext`, registers inputs with
  `RegisterTensor`, sets scalar attrs with `SetAttrAny`, runs the tiling, and
  launches the kernel directly. Tiling data is cached per distinct configuration
  by `TilingTensorCache`.
- **Persistent tiling cache** — a single 512-entry buffer holds the cached
  tiling configs; each config is written once and its address is reused for the
  process lifetime, so graph-captured kernels always read a stable tiling
  address. The buffer is allocated with `aclrtMalloc` (outside the torch caching
  allocator), because allocator-owned memory could be re-mapped/re-written by
  NPU graph capture between the host write and the device read. Once the 512
  entries are full, an eager call falls back to a one-shot `at::empty` buffer
  kept alive with `record_stream` until the current stream drains, while a
  capture-time miss fails with a warmup error (a one-shot address is not safe
  for a captured graph).
- **`readGen` generation handshake** — on arch35 the host zeroes the AIV
  double-buffer release counters before launch (see *arch35-only Additions*).
- **`cache_mode=2` CYCLE ring** — per-request ring state addressed by
  `state_block_table` bank ids; the arch35 vector path uses dedicated
  micro-kernels under `op_kernel/arch35/vf/`.
- **`EMPTY_X` template** — an empty-input call returns immediately without
  launching the compute path; padding/idle graph slots rely on `seqused = 0`
  rather than on `EMPTY_X`.

## MTP / Target Verify Semantics

- `start_pos` must be the committed prefix length; `seqused` is the number of
  draft tokens for the call.
- The ring size must cover the maximum draft width so pending tokens do not
  overwrite state still needed for the committed history.
- Rejected draft suffixes are not rolled back explicitly; the next call
  overwrites them at the same absolute ring slot.
- Graph padding/idle slots must pass `seqused = 0` (and must not carry a stale
  `start_pos`/bank id).

## Constraints

- TH layout, `rotary_mode=2`, `cache_mode=2` for the sglang path.
- `x`, `wkv`, `wgate` must share dtype (`bfloat16`/`float16`); `state_cache` and
  the norm/rope/ape inputs are FP32; metadata tensors are INT32.
- `head_dim` supports 128 and 512; `cmp_ratio` supports 4 and 128 for the DSV4
  variants above.
- Supports graph mode. `state_cache` must be contiguous in its first dimension.

## Usage Example

See [test_compressor.py](../../tests/python/sgl_kernel_npu/test_compressor.py) for
single-operator tests (CPU golden comparison, MTP partial-accept, graph
capture/replay). The sglang integration entry point is
`forward_compress()` in `ascend_dsv4_backend.py`.
