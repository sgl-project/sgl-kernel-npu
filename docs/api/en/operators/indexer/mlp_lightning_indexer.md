# mlp_lightning_indexer

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `auto` / `ascendc` | Supported implementation | Unsupported |
| `torch_native` | Independent PyTorch reference | Unsupported |

This table describes implementation scope, not a new hardware validation result.
In an arch22 Linux/CANN environment, run the root
`python3 setup.py --arch-version 2201 --ops mlp_lightning_indexer`, then install
the generated wheel. `--ops all` on arch22 also includes this operator.
See the [build guide](../../../../../python/sgl_kernel_npu/README.md).

## Signature

```python
from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer

def mlp_lightning_indexer(
    query, key, weights,
    cur_seq_lengths_query=None, cur_seq_lengths_key=None,
    block_table=None, init_tensor=None, local_tensor=None,
    layout_query="BSND", layout_key="PA_BSND", sparse_count=2048,
    kv_block_len=1, q_block_len=1, init_num=0, local_num=0,
    sparse_mode=3, pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807, return_value=False, *, impl="auto",
):
    ...
```

`auto` selects `ascendc`, calling `torch.ops.npu.mlp_lightning_indexer` without
fallback. The native names, defaults, and positional argument order are retained.
The paired LongCat2 caller supplies int64 lengths; this wrapper does not cast them.

## Description

Select key-block indices for LongCat2 sparse attention. For query position `q`,
key position `t`, query head `h`, and key-block width `L = kv_block_len`, compute:

```text
token_score(q,t) = sum_h weights[q,h] * max(dot(query[q,h,:], key[t,0,:]), 0)
block_score(q,b) = sum_{t in block b} token_score(q,t) / L
```

Paged keys are gathered using `block_table`; returned indices are logical block
numbers within the request, not physical page IDs. With `L=1` they are token
indices. Sparse mode 0 exposes the whole key sequence. Mode 3 uses right-aligned
causal visibility: at logical query position `q`, the visible token count is
`key_length - query_length + q + 1`; candidates have block number less than
`ceil(visible_count / L)`.

The initial `init_num / L` blocks and blocks whose start is at or beyond
`visible_count - local_num` receive priority scores; a zero local count disables
local priority. `init_tensor` and
`local_tensor`, when provided, replace the corresponding scalar count per request.
Ordinary scores are capped at `float32_max / 4`, priority scores use
`float32_max / 2`, and the largest `sparse_count` candidates are selected. No extra
diagonal-priority rule is introduced. Equal-score ordering is not an additional
API guarantee.

When `return_value=True`, selected scores are capped at 65504 and cast to the
query dtype, including for bfloat16. Missing candidates have index `-1` and value
`-inf`. When `return_value=False`, the return remains a two-Tensor tuple with the
same full shapes; **the values Tensor's contents are undefined and must not be
consumed or compared**.

For TND, both length arrays are int64 cumulative boundaries of length `B+1`,
including a leading zero. A logical zero-length query request occupies one
physical query/weights/output slot. Thus physical query storage has
`T = sum_r max(query_length[r], 1)` rows; the logical boundaries still use the
unmodified query lengths. The zero-query slot's indices are padding. This is the
existing kernel convention, not a new general-purpose ragged layout.

## Parameters and return value

Tensor inputs follow the existing same-NPU, contiguous caller contract. Q/K have
the same dtype, `N_k=1`, and head dimension `D=128`. These descriptions record the
existing contract; the public dispatch layer adds no new input checks or casts.

| Parameter | Type / shape / dtype | Default | Meaning |
| --- | --- | --- | --- |
| `query` | Tensor, FP16/BF16; TND `[T,N_q,128]`, BSND `[B,S_q,N_q,128]` | Required | Query states; TND includes physical zero-query slots. |
| `key` | Tensor, same dtype as query; PA_BSND `[P,page_size,1,128]`, TND `[T_k,1,128]`, BSND `[B,S_k,1,128]` | Required | Keys. Native page size is a multiple of 16 in `(0,1024]`. |
| `weights` | Tensor, float32; TND `[T,N_q]`, BSND `[B,S_q,N_q]` | Required | Per-query, per-head score weights. |
| `cur_seq_lengths_query` | Optional Tensor, int64; TND `[B+1]` | `None` | Required for TND; logical cumulative query boundaries. No int32 conversion is performed. |
| `cur_seq_lengths_key` | Optional Tensor, int64; TND query path `[B+1]` | `None` | Logical cumulative key boundaries for the TND path; needed with TND or paged keys. |
| `block_table` | Optional Tensor `[B,max_pages]`, int32 | `None` | Logical-to-physical page IDs; required for PA_BSND, absent for non-paged keys. |
| `init_tensor` | Optional Tensor `[B]`, int32 | `None` | Per-request initial token counts, overriding `init_num`. |
| `local_tensor` | Optional Tensor `[B]`, int32 | `None` | Per-request local token counts, overriding `local_num`. |
| `layout_query` | str | `"BSND"` | Historical native layouts: BSND or TND. Reference coverage is described below. |
| `layout_key` | str | `"PA_BSND"` | PA_BSND, or the same non-paged layout as query. |
| `sparse_count` | int | `2048` | Output width; native accepts `1..2048`, or multiples of 128 through 8192. |
| `kv_block_len` | int | `1` | Scoring block width: 1, 2, 4, 8, or 16; distinct from physical page size. |
| `q_block_len` | int | `1` | Existing native implementation requires 1. |
| `init_num` | int | `0` | Initial token count: `0..16`, multiple of `kv_block_len`. |
| `local_num` | int | `0` | Local token count: `0..2048`, multiple of `kv_block_len`. |
| `sparse_mode` | int | `3` | 0: full candidate sequence; 3: right-aligned causal candidates. |
| `pre_tokens` | int | `9223372036854775807` | Retained native attribute; the current kernel does not use it for a window mask. |
| `next_tokens` | int | `9223372036854775807` | Retained native attribute; the current kernel does not use it for a window mask. |
| `return_value` | bool | `False` | Whether score values have defined contents. |
| `impl` | str | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Type / shape / dtype | Meaning |
| --- | --- | --- |
| `indices` | Tensor, int32; TND `[T,1,sparse_count]`, BSND `[B,S_q,1,sparse_count]` | Selected logical key-block numbers; padding is `-1`. |
| `values` | Tensor, query dtype, same shape as indices | Selected scores only when `return_value=True`; otherwise contents are undefined. |

Outputs are new tensors on the query device. Inputs are not modified.

## ascendc backend

Uses the official CANN OpDef, shape/dtype inference, tiling, and template kernel
through this repository's Torch adapter, calling
`aclnnSglMlpLightningIndexerGetWorkspaceSize` and `aclnnSglMlpLightningIndexer`.
CANN manages tiling and kernel dispatch; the private tiling cache from the old
direct-launch path is no longer used. The public API retains the historical BSND
signature and branch without expanding it. TND with PA_BSND is the paired
LongCat2 path; the caller continues to supply int64 lengths.

NPU accuracy, performance, and the existing graph capture/replay cases must be
rerun with the ACLNN build. Earlier direct-launch graph results do not validate
this new execution path.

## torch_native backend

Independently computes scores, block reduction, candidate priorities, and selection
using PyTorch. The current accuracy tests cover TND queries with PA_BSND or TND
keys, including the existing zero-query physical slot convention. The reference
also retains BSND branches and uses explicit `B+1` cumulative boundaries for both
length arrays. The native BSND host expects length arrays of size `B`, while its
device branch reads cumulative boundaries; this inherited discrepancy is outside
the current TND validation scope. The reference does not synthesize lengths when
they are `None`.

This is an accuracy reference, not an automatic native fallback. As with the
native result, do not compare the values output when `return_value=False`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer

device = "npu:0"
torch.npu.set_device(device)
query = torch.randn(1, 8, 128, device=device, dtype=torch.float16)
key = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float16)
weights = torch.ones(1, 8, device=device, dtype=torch.float32)
q_lengths = torch.tensor([0, 1], device=device, dtype=torch.int64)
k_lengths = torch.tensor([0, 16], device=device, dtype=torch.int64)
pages = torch.tensor([[0]], device=device, dtype=torch.int32)
indices, values = mlp_lightning_indexer(
    query, key, weights,
    cur_seq_lengths_query=q_lengths, cur_seq_lengths_key=k_lengths,
    block_table=pages, layout_query="TND", layout_key="PA_BSND",
    sparse_count=8, return_value=True, impl="ascendc",
)
# Both outputs have shape [1, 1, 8]. Select impl="torch_native" for the reference.
```
