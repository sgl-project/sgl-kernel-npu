# DSV4 DSpark tree operators — implementation semantics

Status: Wave 1 implementation brief (source of truth for implementers).
Source pin: **vllm-ascend `8947f1f9f`** ("[Feature] Add tree speculation for qwen3 and dsv4").
Read source with `git show 8947f1f9f:<path>`. Every citation below is
`8947f1f9f:<path>:<line>` (or `:<start>-<end>` for a range). Citations refer to
the commit, not the current working tree.

This document describes the **four AscendC tree operators** that the Python-only
Triton port in `python/sgl_kernel_npu/sgl_kernel_npu/tree/` must reproduce:

| Short name | AscendC operator | Public Python name (planned) |
|---|---|---|
| `project` | `compressor_tree_project` | `compressor_tree_project` |
| `compressor` | `compressor_tree` | `compressor_tree` |
| `indexer` | `quant_lightning_indexer_tree` | `quant_lightning_indexer_tree` |
| `attention` | `sparse_attn_sharedkv_tree` | `sparse_attn_sharedkv_tree` |

The independent CPU oracle lives in `tests/python/sgl_kernel_npu/tree_reference.py`
and is **never** imported by production code (D5).

---

## 0. Locked decisions (D1–D8)

These decisions were fixed before implementation. Do not re-litigate them in a
kernel PR; raise a new decision if source evidence contradicts one.

| # | Decision | Consequence for implementers |
|---|---|---|
| **D1** | **Semantic contract = the AscendC math at `8947f1f9f`, re-expressed as exact PyTorch.** Implementation-specific rounding (per-512-block bf16 probability rounding, fp16 head-reduction order) is documented but **not** required to be bit-reproduced. | Acceptance is tolerance-based vs the independent reference, not bit-parity with the AscendC kernel. |
| **D2** | **`metadata` (`int32[1024]`) is accepted, validated for dtype/shape/contiguity, and otherwise ignored.** Kernel scheduling is self-computed. | A zeroed metadata tensor and a “realistic” metadata tensor must produce identical output. The Triton kernels must **not** read the 36-AIC/72-AIV partition. |
| **D3** | **Top-k tie order is not contractual.** The kernel uses deterministic `(score desc, index asc)`. | Tests compare top-k as a **multiset** and compare score sets; never assert index order between equal scores. |
| **D4** | **`layout_q` defaults to `"TND"`; `"BSND"` raises `NotImplementedError`.** | The tree path is TND-only. |
| **D5** | **References live in `tests/python/sgl_kernel_npu/tree_reference.py`; they import only `torch` and are never imported by production code.** | The reference is independent and runs on a CPU-only box with no `.so`, no Triton, no `torch_npu`. |
| **D6** | **Kernel grid/blocking is static per specialization; `ratio ∈ {4,128}` and `head_dim ∈ {128,512}` become Triton `constexpr` specializations.** | Four specializations per op, mirroring the AscendC tiling keys. |
| **D7** | **`compressor_tree` and `sparse_attn_sharedkv_tree` take `rope_dim`/`cmp_ratio` as runtime args, assert supported values, then dispatch to constexpr kernels.** | Public signatures stay identical to the C++ bindings. |
| **D8** | **No NPU is required to merge Wave 1/2 increments except for explicitly `npu`-marked tests.** | Locally, NPU tests skip with a reason. The numerical GREEN gate is the device command block in §8. |

Two source corrections to the earlier plan are recorded here and must be used:

1. `compressor_tree.prefix` is **`int32`**, not `int64`.
   `8947f1f9f:csrc/attention/compressor_tree/torch_binding.h:27-28` checks
   `{prefix, table}` as `at::kInt`, and the kernel declares
   `GlobalTensor<int32_t> table_, prefix_`
   (`8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:131`).
   `depths`, `requests`, and `ancestors` remain `int64`
   (`8947f1f9f:csrc/attention/compressor_tree/torch_binding.h:29-30`).
2. `compressor_tree_project.x` is **BF16 only at the Python binding boundary**.
   The op definition advertises BF16/FP16
   (`8947f1f9f:csrc/attention/compressor_tree_project/op_host/compressor_tree_project_def.cpp:25-29`),
   but the binding rejects anything except contiguous BF16
   (`8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:10`).
   The port follows the binding.

---

## 1. Common tree model (read this before the op sections)

### 1.1 Nodes, prefixes, depths, requests

- Node `0` is the accepted root. Non-root nodes are `1..n-1`; `n` is the
  flattened number of query records (one query per node). This is the
  **one query per tree endpoint** contract, e.g.
  `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:8-12` and
  `8947f1f9f:csrc/attention/compressor_tree_project/README.md:8-11`.
- `prefix[node]` is that node's base history position; `depth[node]` is its
  tree depth; `pos = prefix + depth`.
- `requests[node]` is the request index for the node.
- `tree_prefix` is the same `prefix` tensor for the attention op; it is
  **`int32`**
  (`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:29-32`,
  `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/torch_binding.h:98-107`).
- `tree_ancestors[node, :]` is **root-first, includes self, and is padded with
  `-1`** (or, at the caller layer, the request root). The vendor kernels must
  never read an unused tail entry
  (`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:31-32`).
- `seqused_kv[node]` is the **logical** prefix length plus ancestor count, not a
  physical cache offset
  (`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:15-18`).

The SGLang host oracle encodes the canonical geometry:

- `positions = prefix + depth` and `node_seqused_kv = prefix + depth + 1`
  (`sglang:python/sglang/srt/speculative/dspark_components/tree/metadata.py:70-72`).
- `request_seqused_kv = prefix_lens + query_width`
  (`sglang:.../metadata.py:95`).
- Sparse original-KV rows are `[prefix window, visible ancestors, -1 padding]`,
  width 128-aligned (`sglang:.../sparse.py:39-46,55-60`).

### 1.2 Ancestor mask semantics (all ops)

| Situation | Required handling |
|---|---|
| `ancestors[node, step] == -1` | Slot is absent: compressor skips it; indexer emits score `-inf` / index `-1`; attention skips it. |
| `ancestors[node, step] >= n` | Same as absent (out-of-range ancestor). Evidence: `8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:110-113`, `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/tree_index.h:15-19`, `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_scfa_block_vector.h:559-563`. |
| `step < 0` or `step >= maxDepth` | Slot is absent; do not index `ancestors`. |
| Position `p < 0` | Compressor/attention skip. Evidence: `8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:105-106`. |
| Position `p > pos` | Compressor/attention skip (future position). |
| No valid key at all | Compressor emits an all-zero row; attention emits `out=0, lse=0` (exact, not sink). |
| Unused ancestor tail | Never read; the root-first array is only indexed by `0..step`. |

### 1.3 Logical position → physical source (C4 / C128)

Both compressed ops share the same mapping. For a logical compressed index `k`
and node row with `prefix = tree_prefix[row]`, `ratio = cmp_ratio`:

```
if k < prefix // ratio:
    page = table[requests[row], k // block]
    slot = page * block + (k % block)          # paged history
else:
    step = (k + 1) * ratio - 1 - prefix
    node = ancestors[row, step]                # root-first, -1 padded
    if step outside [0, width) or node outside [0, n):
        absent (score -inf / index -1)
    else:
        source = tree row / node KV
```

Evidence: `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/tree_index.h:15-19`
(indexer) and
`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_scfa_block_vector.h:542-583`
(attention, `GetKeyGmOffset`). For attention the tree branch returns
`cmpCacheSpan + node * headDim` (line 565), i.e. offsets above the paged cache
span select the node pool.

C4 (`ratio = 4`) interleaves the previous/current windows using two projected
halves; C128 (`ratio = 128`) uses one window. Evidence:
`8947f1f9f:csrc/attention/compressor_tree/README.md:8-14` and
`8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:42-53`.

### 1.4 Mask modes

- `0 = DEFAULT_MASK` (no mask).
- `3 = RIGHT_DOWN_CAUSAL`.
- `4 = BAND`.

Only `ori_mask_mode = 4` and `cmp_mask_mode = 3` are legal for tree attention;
reject other values. Evidence:
`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/torch_binding.h:116-126`
(defaults 4/3; `ori_win_left` default 127) and
`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:23-28`.

### 1.5 Metadata: validated but ignored (D2)

Both `quant_lightning_indexer_tree` and `sparse_attn_sharedkv_tree` accept an
optional `metadata` tensor of **`int32[1024]`**. The layout is identical:

- `AIC_CORE_MAX_NUM / AIC_CORE_NUM = 36`, `AIV_CORE_MAX_NUM / AIV_CORE_NUM = 72`,
  total `1024` int32 entries.
- Per AIC core: 8 ints (`CORE_ENABLE`, `BN2_START`, `M_START`, `S2_START`,
  `BN2_END`, `M_END`, `S2_END`, `FIRST_*_WORKSPACE_IDX`).
- Per AIV core: 8 ints (`CORE_ENABLE`, `BN2_IDX`, `M_IDX`, `WORKSPACE_IDX`,
  `WORKSPACE_NUM`, `M_START`, `M_NUM`; the 8th slot is reserved).
- Absolute AIC index = `8 * coreIdx + metaIdx`; absolute AIV index =
  `8 * 36 + 8 * coreIdx + metaIdx`.

Evidence:
`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/quant_lightning_indexer_tree_metadata.h:24-64`
and
`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/sparse_attn_sharedkv_tree_metadata.h:24-64`.

**D2 rule:** validate `dtype == int32`, `numel == 1024`, and `is_contiguous()`;
then ignore the contents. Do not couple Triton launch geometry to this table.

---

## 2. `compressor_tree_project` (project)

### 2.1 Signature

```text
compressor_tree_project(x, wkv, wgate, nodes, ape, positions, cu) -> None
```

The `nodes` argument is **in-place output**. There is no separate output tensor.
Evidence: `8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:43-48`.

### 2.2 Inputs

| Tensor | Dtype | Shape | Strides / constraints |
|---|---|---|---|
| `x` | BF16 | `[T, H]` | contiguous; `H` is the hidden size (`1024..10240`, multiple of 512 in tests) |
| `wkv` | BF16 | `[c*d, H]` | contiguous; `c = 2` for `ratio=4`, else `1`; `d = head_dim` |
| `wgate` | BF16 | `[c*d, H]` | contiguous; same shape rules |
| `nodes` | FP32 | `[T+1, 1, 2*c*d]` (numel `(T+1)*2*c*d`) | contiguous; **row 0 is reserved and must not be written** |
| `ape` | FP32 | `[ratio, c*d]` | contiguous; `ratio ∈ {4,128}`; `d = ape.size(1)/c ∈ {128,512}` |
| `positions` | int32 | `[T]` | contiguous; APE residue `(prefix + depth) % ratio` |
| `cu` | int32 | `[T+1]` | contiguous; **must equal `arange(T+1)`** (one independent query row per node) |

Evidence for these checks:
`8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:10-23`.
The “one query per node” / `cu = arange(n+1)` contract is also stated in
`8947f1f9f:csrc/attention/compressor_tree_project/README.md:8-11`.
`nodes` is viewed as `[T+1,1,2*c*d]` in the binding
(`8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:28-29`).
The op definition lists BF16/FP16 for `x/wkv/wgate`
(`8947f1f9f:csrc/attention/compressor_tree_project/op_host/compressor_tree_project_def.cpp:25-39`),
but the binding enforces BF16 (see §0 correction).

### 2.3 Host constraints (fail fast)

Reject before launching:

- `x.ndim != 2` or `x.dtype != bf16` or not contiguous.
- `ape.ndim != 2` or `ape.size(0) not in {4,128}`.
- `d = ape.size(1) / c` not in `{128,512}`.
- `wkv`/`wgate` shape not `[c*d, H]`, wrong dtype/device, or non-contiguous.
- `nodes` wrong dtype/numel/device or non-contiguous.
- `positions`/`cu` wrong rank/numel/dtype/device or non-contiguous.
- `cu != arange(T+1)` (the kernel assumes `cu[node+1] - cu[node] == 1`).
- `positions` values outside `[0, ratio)`.

### 2.4 Algorithm

For each node index `i` in `[0, T)` (row `i+1` of `nodes`):

```
kv      = x[i] @ wkv.T                     # FP32 accumulate
score   = x[i] @ wgate.T + ape[r, :]       # r = positions[i]; APE on score only
nodes[i+1] = [ kv[0:c*d] | score[0:c*d] ]  # c*d each; half order = previous/current
```

`nodes[0]` is untouched. The kernel never aggregates a linear window, never runs
RMSNorm/RoPE, and never writes `cmp`; those placeholders in the CANN ABI are
dead on this path
(`8947f1f9f:csrc/attention/compressor_tree_project/README.md:3-19`).
The half ordering is `coff = c` halves for KV followed by `c` halves for score:
`8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:123-126`
loads KV at `offset + half*d` and score at `offset + c*d + half*d`.

### 2.5 Triton port shape (planning only)

- Grid `(ceil(T/BM), 2*c*d/BN)` with `BM=16`, `BN=128`.
- K-loop over `H` in `BK=256`; `tl.dot` BF16×BF16→FP32.
- APE add fused for columns `>= c*d` using `ape[r, col-c*d]`.
- Store FP32; mask `node < T`; write row `node+1` only.

---

## 3. `compressor_tree` (compressor)

### 3.1 Signature

```text
compressor_tree(state, nodes, norm, cos, sin, table,
                prefix, ancestors, depths, requests,
                ratio, rope_dim, epsilon) -> Tensor[n, d] BF16
```

Evidence: `8947f1f9f:csrc/attention/compressor_tree/torch_binding.h:47-50`.

### 3.2 Inputs

| Tensor | Dtype | Shape | Strides / constraints |
|---|---|---|---|
| `state` | FP32 | `[pages, rows, ..., 2*c*d]` | `stride(1) >= 2*c*d` (padded rows allowed); not modified |
| `nodes` | FP32 | `[(n+1)*2*c*d]` (viewed `[n+1,1,2*c*d]`) | contiguous; row 0 zero/reserved |
| `norm` | FP32 | `[d]` | contiguous; `d ∈ {128,512}` |
| `cos` | FP32 | `[..., >=rope_dim]` | `stride(1) == 1`; row stride may exceed `rope_dim` |
| `sin` | FP32 | same as `cos` | same |
| `table` | int32 | `[requests, width]` | contiguous 2D page table |
| `prefix` | **int32** | `[n]` | contiguous (see §0 correction) |
| `ancestors` | int64 | `[n, maxDepth]` | contiguous; root-first, `-1` padded |
| `depths` | int64 | `[n]` | contiguous |
| `requests` | int64 | `[n]` | contiguous |

Attrs: `ratio ∈ {4,128}`, `rope_dim == 64`, `epsilon > 0`.
Evidence: `8947f1f9f:csrc/attention/compressor_tree/torch_binding.h:16-33`.
The supported `head_dim 128/512, ratio 4/128, rope 64` set is stated in
`8947f1f9f:csrc/attention/compressor_tree/README.md:22-23`.
The kernel declares `prefix_` and `table_` as `int32` and
`ancestors_/depths_/requests_` as `int64`
(`8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:131-132`).

### 3.3 Host constraints

Reject before launching:

- `ratio not in {4,128}`; `rope_dim != 64`; `epsilon <= 0`.
- `norm.ndim != 1` or `norm.size(0) not in {128,512}`.
- `state.ndim < 3`, `state.size(1) <= 0`, or `state.stride(1) < 2*c*d`.
- `nodes` not contiguous or `numel != (n+1)*2*c*d`.
- `table` not 2D/contiguous; `prefix` not int32/contiguous; `ancestors/depths/requests` not int64/contiguous.
- `prefix.numel() != n`, `depths.numel() != n`, `requests.numel() != n`.
- `cos/sin` not 2D, `size(1) < rope_dim`, or `stride(1) != 1`.
- State inputs are read-only: never write `state`, `nodes`, `cos`, `sin`.
  Evidence: `8947f1f9f:csrc/attention/compressor_tree/README.md:8-18`.

### 3.4 Algorithm

For each node `i`:

```
start = prefix[i]; depth = depths[i]; pos = start + depth
emit iff (pos + 1) % ratio == 0
if not emit: output row = all zeros (exact)
for w in [0, ratio*c):
    p    = pos + 1 - ratio*c + w//c + (w % c) * ratio
    half = w % c
    if p < 0 or p > pos: continue
    if p >= start:
        step = p - start
        if step >= maxDepth: continue
        a = ancestors[i, step]
        if a < 0 or a >= n: continue
        kv    = nodes[(a+1)*2*c*d + half*d : ... + d]
        score = nodes[(a+1)*2*c*d + c*d + half*d : ... + d]
    else:
        col = p // rows
        if col >= width: continue
        page = table[requests[i], col]
        if page < 0: continue
        kv    = state[page, p % rows, half*d : ... + d]
        score = state[page, p % rows, c*d + half*d : ... + d]
    m = max(m, score)
    S += exp(score - m)
    V += exp(score - m) * kv
V /= S
V /= sqrt(mean(V*V) + eps)     # eps INSIDE sqrt
V *= norm
ropePos = pos + 1 - ratio
for i in [0, rope_dim):
    j = i ^ 1
    sign = -1 if i even else +1
    out[d-rope_dim+i] = V[d-rope_dim+i]*cos[ropePos,i] + sign*V[d-rope_dim+j]*sin[ropePos,i]
output -> BF16, non-emitting nodes -> zero row
```

Evidence for every step:
`8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:44-100`
(emit condition line 47, max init line 48, interleaving lines 50-53, max pass
lines 55-60, exp/weighted sum lines 61-72, divide line 73, RMSNorm lines 74-81,
RoPE pairing/sign/ropePos lines 82-96, BF16 cast line 98).
The public README confirms “non-emitting nodes produce zeros”, “negative/unused
ancestor slots and positions before the sequence start are masked”, and
“state inputs are never modified”
(`8947f1f9f:csrc/attention/compressor_tree/README.md:8-18`).

### 3.5 Triton port shape (planning only)

- Grid `(n, d/BD)` with `BD=128` (`d=512` → 4 programs per node), or one program per node.
- Compile-time `(ratio, c)`; runtime `rope_dim` asserted to 64.
- Two passes: max over valid slots, then `exp/sum/weighted sum`.
- `i ^ 1` gather, sign `-1/+1`, `ropePos = pos + 1 - ratio`.
- Non-emitting rows are exactly zero; output BF16.

---

## 4. `quant_lightning_indexer_tree` (indexer)

### 4.1 Signature

```text
quant_lightning_indexer_tree(
    query, history, weights, query_scale, history_scale,
    nodes, node_scale, ancestors, prefixes, requests,
    table, cu, lengths, residual, metadata, topk
) -> int32 [n, 1, topk]
```

The binding names are `quant_lightning_indexer_tree(query, history, weights,
query_scale, history_scale, nodes, node_scale, ancestors, prefixes, requests,
table, cu, lengths, residual, metadata, topk)`.
Evidence: `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/torch_binding.h:62-69`.

### 4.2 Inputs

| Tensor | Dtype | Shape | Strides / constraints |
|---|---|---|---|
| `query` | int8 | `[n, 64, 128]` | contiguous; `n > 0`; exactly 64 query heads |
| `history` | int8 | `[pages, block, 1, 128]` | `stride(1) == 128`, `stride(3) == 1` (padded page stride allowed) |
| `history_scale` | FP16 | `[pages, block, 1, 1]` | `stride(1) == 1`; same pages/block as history |
| `weights` | FP16 | `[n, 64]` | contiguous |
| `query_scale` | FP16 | `[n, 64]` | contiguous |
| `nodes` (tree key) | int8 | `[n, 128]` | contiguous |
| `node_scale` | FP16 | `[n]` | contiguous |
| `ancestors` | int64 | `[n, >0]` | contiguous; root-first, `-1` padded |
| `prefixes` | int32 | `[n]` | contiguous |
| `requests` | int64 | `[n]` | contiguous |
| `table` | int32 | `[reqs, width]` | contiguous 2D |
| `cu` | int32 | `[n+1]` | contiguous; `cu = arange(n+1)` |
| `lengths` | int32 | `[n]` | contiguous |
| `residual` | int32 | `[n]` | contiguous |
| `metadata` | int32 | `[1024]` | contiguous; validated, ignored (D2) |
| `topk` | int | — | `1 <= topk <= 2048` |

Evidence: `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/torch_binding.h:20-34`
(shape/dtype/stride checks), `:35-43` (`topk` range, contiguity/colocation).
Supported contract: A2/A3, INT8 Q/K, FP16 scales/weights, 64 query heads,
head dimension 128, ratio 4, TND queries, paged history
(`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:14-18`).
The op definition lists `tree_key` int8, `tree_scale` FP16, `tree_ancestors`
int64, `tree_prefix` int32, `tree_requests` int64, and `topk` default 2048
(`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_host/quant_lightning_indexer_tree_def.cpp:81-99`).
`head_dim == 128` is enforced by `HEAD_DIM_LIMIT`
(`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_host/quant_lightning_indexer_tree_tiling.h:89`).

### 4.3 Algorithm

For row `row`, query head `g`:

```
QK[g] = int8_dot(query[row,g,:], k[pos,:])          # int32, exact
S[g]  = fp16_rne(relu(QK[g]) * 2^-10)               # 0x3a800000, reluPre=1
Wq[g] = fp16_rne(weights[row,g] * query_scale[row,g])
P     = sum_g float(S[g]) * float(Wq[g])            # FP32
score = P * float(history_scale[pos])               # FP32
```

No softmax, no `sqrt(d)`, no post-activation.
Evidence: `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/arch22/quant_lightning_indexer_tree_service_cube_arch22.h:557-561`
(`DEQF16`, `reluPre = 1`, `SetFixpipePreQuantFlag(0x3a800000)`); the FP16
rounding boundary and head reduction are described in
`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:3-6`.

Position resolution:

```
historyEnd = prefixes[row] // 4
if k < historyEnd:
    page  = table[requests[row], k // block]
    key   = history[page, k % block, 0, :]
    scale = history_scale[page, k % block, 0, 0]
else:
    step = (k + 1) * 4 - 1 - prefixes[row]
    if step < 0 or step >= ancestor_width: invalid
    node = ancestors[row, step]
    if node < 0 or node >= n: invalid
    key   = nodes[node]
    scale = node_scale[node]
invalid -> score = -inf (0xFF800000), index = -1 before sorting
```

Evidence: `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/tree_index.h:15-19`
(`step = (k+1)*4-1-prefix`; `node` bounds), `8947f1f9f:.../quant_lightning_indexer_tree_vector.h:24`
(`NEG_INF = 0xFF800000`), and the kernel’s absent-ancestor rejection
`8947f1f9f:.../quant_lightning_indexer_tree_service_vector_arch22.h:875-882`.
The README states absent ancestors get negative infinity and index `-1` before
sorting (`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:8-12`).

Top-k:

- Per-row `2048`-wide `(-inf,-1)` accumulator, tile sort (Sort32 + 4-way MrgSort) then 2-way merge; descending by score.
- Output `[n,1,topk]` int32; valid indices first, `-1` padding.
- `sparse_values` is empty (`return_value=0`).
- Deterministic `(score desc, index asc)` for equal scores; tie order is not contractual (D3).

Evidence: `8947f1f9f:.../quant_lightning_indexer_tree_service_vector_arch22.h:31-32`
(`BASE_TOPK = 2048`, `BASE_TOPK_VALUE_IDX_SIZE = 4096`), `:266-267` (`-inf,-1`
initialization), `:894-897` (sort + merge), `:875-882` (absent ancestor).
`return_value=0` is passed from
`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/torch_binding.h:50-57`.

Select-all shortcut:

```
if max(1, max_kv // 4) <= topk:
    return chronological arange(topk), masked by (start + depth + 1)//4, -1 pad
    # never call the scoring kernel
```

This is Python-side only (`native_indexer.py`), documented in
`8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:31-35`.

### 4.4 Triton port shape (planning only)

- `_gather_kernel`: grid `(n, ceil(kv/BLOCK_KV))`, `BLOCK_KV=128`; resolve
  paged-history / tree-ancestor indirection once and materialise contiguous
  `[n, slots, 128]` int8 keys plus per-slot scale and validity.
- `_score_kernel`: grid `(n, ceil(kv/BLOCK_KV))`; read the contiguous key
  matrix, int32-widening fallback (or optional int8 `tl.dot` -> int32),
  `relu`, `*2^-10`, FP16 cast, `w*qs` in FP16, FP32 head reduce, `*scale`.
  The previous fused kernel re-gathered the key row for every head and D
  chunk (`64 * 16` times), which dominated the A3 fallback.
- `_topk_merge_kernel`: reuse the 2048-wide accumulator / merge pattern from
  `indexer/flash_block_score_decode.py`; `(score desc, index asc)`.
- Validate `topk in [1,2048]`.

---

## 5. `sparse_attn_sharedkv_tree` (attention)

### 5.1 Signature

```text
sparse_attn_sharedkv_tree(
    q,
    *,
    ori_kv=None, cmp_kv=None,
    ori_sparse_indices=None, cmp_sparse_indices=None,
    ori_block_table=None, cmp_block_table=None,
    cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
    seqused_q=None, seqused_kv=None,
    sinks=None, metadata=None,
    cmp_tree_kv=None, tree_ancestors=None, tree_prefix=None,
    softmax_scale=0, cmp_ratio=0,
    ori_mask_mode=4, cmp_mask_mode=3,
    ori_win_left=128, ori_win_right=0,
    layout_q="BSND", layout_kv="PA_ND",
    return_softmax_lse=False
) -> (out, softmax_lse)
```

For the tree path, callers pass `layout_q="TND"`, `layout_kv="PA_ND"`,
`ori_mask_mode=4`, `cmp_mask_mode=3`.
Evidence: `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/torch_binding.h:92-128`.

### 5.2 Inputs

| Tensor | Dtype | Shape | Strides / constraints |
|---|---|---|---|
| `q` | FP16/BF16 | `[T, N1, 512]` | TND; all dims > 0; `head_dim == 512` |
| `ori_kv` | FP16/BF16 | `[pages, block, N2, 512]` | paged; `N2 == 1`; `stride(0)` passed as `ori_kv_stride` |
| `cmp_kv` | FP16/BF16 | `[pages, block, N2, 512]` | paged; `N2 == 1` |
| `ori_sparse_indices` | int32 | `[T, 1, W]` | physical original-KV slots; `W` multiple of 128; `-1` padding |
| `cmp_sparse_indices` | int32 | `[T, 1, K]` | C4 only; logical compressed Top-K; `K` 512/1024; `-1` padding |
| `ori_block_table` / `cmp_block_table` | int32 | `[reqs, width]` | paged lookup tables |
| `cu_seqlens_q` | int32 | `[T+1]` | TND query offsets |
| `cu_seqlens_ori_kv` / `cu_seqlens_cmp_kv` | int32 | `[T+1]` | optional; logical lengths |
| `seqused_q` / `seqused_kv` | int32 | `[T]` | `seqused_kv` = logical prefix + ancestor count |
| `sinks` | FP32 | `[N1]` | optional; zero-value key with initial weight 1 |
| `metadata` | int32 | `[1024]` | validated, ignored (D2) |
| `cmp_tree_kv` | FP16/BF16 | `[nodes, 512]` | contiguous; node KV pool |
| `tree_ancestors` | int64 | `[T, max_depth]` | root-first, unused tail never read |
| `tree_prefix` | int32 | `[T]` | per-node prefix base |

Evidence: `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:11-41`
(tree contract), `:29-35` (trio, dtypes, ancestor/prefix), and
`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_host/sparse_attn_sharedkv_tree_def.cpp:23-126`
(dtype/attr declarations). The output copies the query shape; `softmax_lse` is
`[...,1]` FP32 when requested, otherwise `[0]`
(`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_host/sparse_attn_sharedkv_tree_proto.cpp:26-60`).

### 5.3 Host constraints

- `layout_q == "TND"`; `layout_kv == "PA_ND"` (D4).
- `N2 == 1`; `head_dim == 512`.
- `ori_mask_mode == 4`; `cmp_mask_mode == 3`.
- Tree pool inputs (`cmp_tree_kv`, `tree_ancestors`, `tree_prefix`) are
  **all-or-none**; partial supply raises. Evidence:
  `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:29-32`,
  `sglang:.../tree/attention.py:159-170`.
- `ori_sparse_indices` width is a multiple of 128; valid slots are the leading
  entries before the first `-1`, capped by `W` and `seqused`.
  `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:16-20`.
- `cmp_sparse_indices` (C4) must supply `min(topk_width, seqused_kv // 4)` valid
  positions followed by padding; C128 omits it.
  `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:23-28`.
- State/cache inputs are read-only; QK and PV must consume identical original-KV
  slots (`8947f1f9f:.../README.md:40-41`).

### 5.4 Algorithm

```
keys = ori_slots ++ cmp_slots              # original first, then compressed
ori_slots: leading valid entries of ori_sparse_indices[q, 0, :]
           (stop at -1; cap by W and seqused); physical slot s -> ori_kv[s]
cmp_slots (C4/SCFA):
    for k in [0, K):
        logical = cmp_sparse_indices[q, 0, k]
        resolve logical -> paged cmp_kv or cmp_tree_kv[node]
        invalid (logical < 0, absent ancestor) -> skip
cmp_slots (C128/CFA):
    for k in [0, seqused_kv[row] // 128):
        if k < prefix // 128: paged cmp_kv
        else: node = tree_ancestors[row, step] with step = (k+1)*128-1-prefix
SWA (cmp_kv is None): ori only

s = qk * softmax_scale
M0 = sinks[h]; S0 = 1
online: M = max(M, rowmax(s))
        S = S*exp(M_old - M) + sum(exp(s - M))
        Acc = Acc*exp(M_old - M) + (exp(s - M) -> input dtype) @ V
out = Acc / S; LSE = ln(S) + M
empty row: out = 0, lse = 0 (exact, not sink)
```

Evidence: `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_scfa_block_vector.h:542-583`
(`GetKeyGmOffset`), `:389-422` (`ProcessLse`), `:487` (`CAST_ROUND` to KV dtype),
`:954-978` (rescale + `RowDivs`);
`8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_swa_block_cube.h:53-69`
(C128 history/tree split); `8947f1f9f:.../sparse_attn_sharedkv_tree.cpp:50`
(`MIX_AIC_1_2`). The mask-mode meanings are `0 = DEFAULT_MASK`,
`3 = RIGHT_DOWN_CAUSAL`, `4 = BAND` (see §1.4).

### 5.5 Triton port shape (planning only)

- Grid `(T, N1/16, 512/128)`; per program build the slot list on device.
- Loop chunks of 128 slots; gather K/V; `tl.dot` FP16/BF16→FP32.
- Online softmax seeded with `sinks[h]`, `S0=1`; probabilities cast to input
  dtype before PV; final `Acc/S`; `LSE = ln(S)+M`.
- `return_softmax_lse=True` → `[...,1]` FP32.
- Reject `N2 != 1`, `head_dim != 512`, unsupported layouts/masks, partial trio.

---

## 6. Dtype / stride / paging policy (all ops)

| Concern | Rule |
|---|---|
| Activations | BF16 (project), FP16/BF16 (compressor output is BF16; attention output matches Q dtype) |
| Index/table tensors | int32 for prefix/table/cu/lengths/residual/metadata; int64 for ancestors/depths/requests |
| Node state | FP32 `[n+1, 1, 2*c*d]`; row 0 reserved/zero |
| Paged state rows | `stride(1)` may exceed `2*c*d`; pass the stride explicitly |
| Paged indexer history | `stride(1) == 128`, `stride(3) == 1`; page stride may be padded |
| Paged attention KV | page stride passed as `ori_kv_stride`/`cmp_kv_stride`; `N2 == 1` |
| Sparse indices | original physical slots, width multiple of 128, `-1` pad; compressed are logical positions |
| Ancestors | root-first, include self, `-1` pad; unused tail never read |
| RoPE | `rope_dim = 64`, pairing `i ^ 1`, signs `-1,+1`, `ropePos = pos + 1 - ratio` |
| Top-k | `(score desc, index asc)`; ties not contractual; `-1` padding |
| Metadata | int32[1024], validated, ignored (D2) |

---

## 7. Independent reference layer (D5)

`tests/python/sgl_kernel_npu/tree_reference.py` must expose exactly these pure
torch functions (no Triton, no `torch_npu`, no `sgl_kernel_npu` import):

```python
compressor_tree_project_ref(x, wkv, wgate, nodes, ape, positions, cu, ratio, head_dim) -> None
compressor_tree_ref(state, nodes, norm, cos, sin, table, prefix, ancestors,
                    depths, requests, ratio, rope_dim, eps) -> Tensor[n, d]  # bf16
quant_lightning_indexer_tree_ref(query, history, weights, qs, tree_key, tree_scale,
                                 ancestors, prefixes, reqs, block_table, cu,
                                 lengths, residual, metadata, topk, head_dim=128) -> Tensor[n,1,topk]  # int32
sparse_attn_sharedkv_tree_ref(q, ori_kv, cmp_kv, ori_sparse_indices,
                              cmp_sparse_indices, ori_block_table, cmp_block_table,
                              cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                              seqused_q, seqused_kv, sinks, metadata, cmp_tree_kv,
                              tree_ancestors, tree_prefix, softmax_scale, cmp_ratio,
                              ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                              layout_q="TND", layout_kv="PA_ND",
                              return_softmax_lse=False) -> (out, lse)
```

The benchmark imports this module via an explicit `sys.path` bootstrap and runs
it directly on CPU for `--dry-run`. The reference is **not** a fallback for a
missing kernel in production (D5); it is an independent oracle.

---

## 8. Verification commands (Wave 1)

CPU-only (this Windows box, no Triton/NPU):

```bash
python -m py_compile python/sgl_kernel_npu/sgl_kernel_npu/tree/_common.py \
  python/sgl_kernel_npu/sgl_kernel_npu/tree/__init__.py \
  benchmark/spec_decode/bench_tree_ops.py

python benchmark/spec_decode/bench_tree_ops.py --op all --dry-run \
  --json /tmp/tree_dry.json --csv /tmp/tree_dry.csv
```

Reference self-tests (owned by the reference agent):

```bash
python -m pytest tests/python/sgl_kernel_npu/test_tree_reference.py -q
```

NPU / A3 (the only numerical GREEN gate):

```bash
python -m pytest tests/python/sgl_kernel_npu/test_tree_*.py -q
bash scripts/run_kernel_tests.sh speculative
python benchmark/spec_decode/bench_tree_ops.py --op all --warmup 10 --reps 50 \
  --csv perf/tree_perf.csv --json perf/tree_perf.json
```

Wave 1 non-goals: no Triton kernel, no `csrc/CMake`, no
`scripts/run_kernel_tests.sh` edit, no production import of the reference, no
fabricated device measurements.

---

## 9. Known risks / open items

| # | Risk | Mitigation |
|---|---|---|
| R1 | C4/C128 half interleaving (`half = w%c`, `p` formula) silently reads the wrong KV half. | Encode §3.4 exactly in the reference; dedicated poison fixture with `half0=+1/half1=-1`. |
| R2 | Ancestor mask (`-1`, `>=n`, `step>=maxDepth`) produces garbage/NaN. | Mask per §1.2; assert non-emit rows are exactly zero and outputs are finite. |
| R3 | Metadata offsets consumed accidentally. | D2: zeroed vs realistic metadata must produce identical output. |
| R4 | Padded strides (`state.stride(1)`, `history.stride(0)`, `cos/sin` row stride) ignored. | Pass strides explicitly; tests with `+0/+8/+64` padding. |
| R5 | Top-k ties / merge order flakiness. | Compare multisets and score sets; deterministic `(score desc, index asc)`. |
| R6 | RoPE pairing/sign/`ropePos` drift. | Hand-computed 2-element golden; `ropePos` lag test at a ratio boundary. |
| R7 | No A3 locally. | D8: CPU-first gates; NPU parity is the deferred GREEN gate. |
| R8 | Triton Ascend int8 `tl.dot` restrictions. | Probe in Wave 3 RED; fallback to int32 widening + `tl.sum` with documented perf hit. |
| R9 | `d=512`, `N1=64`, `ratio=128` UB/compile pressure. | `BD=128`, `N1` block 16, `BS2=512`; split `d` into two passes if needed. |
| R10 | Import graph loads Triton/`.so` too early. | `tree/__init__.py` is lazy; `tree_reference.py` imports only `torch`. |

---

## Appendix A — Citation index

The following commit-pinned locations are the primary evidence for this brief.
All use the `8947f1f9f:<path>:<line>` form.

- `8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:10-23` — project dtype/contiguity/one-query-per-node checks.
- `8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:43-48` — project signature.
- `8947f1f9f:csrc/attention/compressor_tree_project/README.md:8-11` — `cu = arange(n+1)`, row 0 reserved.
- `8947f1f9f:csrc/attention/compressor_tree_project/op_host/compressor_tree_project_def.cpp:25-39` — BF16/FP16 declaration.
- `8947f1f9f:csrc/attention/compressor_tree/torch_binding.h:16-33` — compressor dtype/stride/ancestor checks.
- `8947f1f9f:csrc/attention/compressor_tree/torch_binding.h:47-50` — compressor signature.
- `8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:44-100` — emit, max/exp/sum, RMSNorm, RoPE.
- `8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:105-126` — address resolution and `[kv|score]` layout.
- `8947f1f9f:csrc/attention/compressor_tree/op_kernel/compressor_tree.cpp:131-132` — int32 prefix/table, int64 ancestors/depths/requests.
- `8947f1f9f:csrc/attention/compressor_tree/README.md:8-18` — masking, zeros, read-only state.
- `8947f1f9f:csrc/attention/compressor_tree/README.md:22-23` — head dims/ratios/rope.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/torch_binding.h:20-43` — indexer shapes/dtypes/topk.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/torch_binding.h:50-69` — `return_value=0`, signature.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/tree_index.h:15-19` — `(k+1)*4-1-prefix` mapping.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/arch22/quant_lightning_indexer_tree_service_cube_arch22.h:557-561` — `DEQF16`, `reluPre=1`, `0x3a800000`.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/arch22/quant_lightning_indexer_tree_service_vector_arch22.h:31-32` — 2048 top-k accumulator.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/arch22/quant_lightning_indexer_tree_service_vector_arch22.h:875-897` — absent-ancestor `-inf/-1`, sort/merge.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/arch22/quant_lightning_indexer_tree_vector.h:24` — `NEG_INF=0xFF800000`.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:8-18` — one query per endpoint, absent ancestors, supported contract.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/README.md:31-35` — select-all shortcut.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_kernel/quant_lightning_indexer_tree_metadata.h:24-64` — 36/72 metadata layout.
- `8947f1f9f:csrc/attention/quant_lightning_indexer_tree/op_host/quant_lightning_indexer_tree_tiling.h:89` — `HEAD_DIM_LIMIT=128`.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/torch_binding.h:92-128` — attention signature/defaults.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/torch_binding.h:116-126` — mask/window defaults.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_host/sparse_attn_sharedkv_tree_def.cpp:23-126` — dtype/attr declarations.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_host/sparse_attn_sharedkv_tree_proto.cpp:26-60` — output/lse shapes.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_scfa_block_vector.h:542-583` — `GetKeyGmOffset`.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_scfa_block_vector.h:389-422` — `ProcessLse`.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_scfa_block_vector.h:954-978` — rescale/`RowDivs`.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/arch32/sparse_attn_sharedkv_tree_swa_block_cube.h:53-69` — C128 history/tree split.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/sparse_attn_sharedkv_tree.cpp:50` — `MIX_AIC_1_2`.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/op_kernel/sparse_attn_sharedkv_tree_metadata.h:24-64` — attention metadata layout.
- `8947f1f9f:csrc/attention/sparse_attn_sharedkv_tree/README.md:11-41` — tree attention contract.
