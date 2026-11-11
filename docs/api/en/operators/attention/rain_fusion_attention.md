# rain_fusion_attention

RainFusionAttention block-sparse fused attention operator.

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `cann_ops_transformer` | Supported | Unsupported |
| `torch_native` | Supported | Unsupported |

## Signature

```python
from sgl_kernel_npu.operators.attention import rain_fusion_attention

def rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    blockshape,
    attn_mask=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    block_table=None,
    q_input_layout="TND",
    kv_input_layout="TND",
    head_num=1,
    mask_type=0,
    scale=1.0,
    inner_precise=1,
    block_size=0,
    impl="auto",
):
    ...
```

`auto` selects `cann_ops_transformer` on arch22. There is no fallback: requesting the operator
on arch35 raises instead of switching backends. The legacy name
`rainfusionattention` is exported as an alias.

## Description

Computes block-sparse attention over the KV blocks selected per query block.
Only the selected blocks take part in the `query @ key^T`, softmax and
`probs @ value` computation:

```text
attention_out = softmax(scale * (query @ key^T)) @ value
```

The sparse pattern comes from `select_idx`/`select_num_idx`; the block tile is
given by `blockshape`.

## Parameters and return values

`B` is the batch size, `N` the head count, `S` the query sequence length and `D`
the head width.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `query` | TND `[T, N, D]` or BNSD `[B, N, S, D]` | FP16 / BF16 | Layout selected by `q_input_layout`. |
| `key`, `value` | TND `[Tkv, Nkv, D]` or BNSD | Same as `query` | Layout selected by `kv_input_layout`. |
| `select_idx` | `[q_blocks, N, kv_blocks]` | INT64 | Valid KV block ids first in ascending order, `-1` padding. |
| `select_num_idx` | `[q_blocks, N]` | INT64 | Number of valid KV blocks per query block and head. |
| `blockshape` | `[block_rows, block_cols]` | Required | Host integer tile size. |
| `attn_mask` | — | `None` | Forwarded as an optional ACLNN tensor; not validated here. |
| `actual_seq_qlen`, `actual_seq_kvlen` | `[B]` | `None` | Host integer sequence boundaries for packed TND inputs. |
| `block_table` | — | `None` | Forwarded as an optional ACLNN tensor for paged inputs. |
| `q_input_layout`, `kv_input_layout` | — | str, `"TND"` | Only `TND` and `BNSD` are accepted; anything else raises. |
| `head_num` | — | int, 1 | Passed to the ACLNN `numKeyValueHeads` slot unchanged. |
| `mask_type` | — | int, 0 | ACLNN mask type. |
| `scale` | — | float, 1.0 | Score multiplier, typically `D ** -0.5`. |
| `inner_precise` | — | int, 1 | Softmax precision mode. |
| `block_size` | — | int, 0 | Paged-attention block size; 0 leaves it unused. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `cann_ops_transformer`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `attention_out` | Same as `query` | Same as `query` |
| `softmax_lse` | The first three `query` dimensions | Same as `query` |

## cann_ops_transformer backend

- Calls the CANN toolkit operator `aclnnRainFusionAttention`. Symbols resolve in
  the shared order: the wheel's `libcust_opapi.so` first, then the CANN toolkit
  `libopapi.so`. No vendor package or extra `ASCEND_CUSTOM_OPP_PATH` entry is
  needed.
- Both outputs are allocated with the query options and the query NPU format
  before the launch, matching the legacy binding.
- `softmax_lse` is allocated for signature compatibility. The current CANN
  operator may leave it untouched, so treat its content as undefined and
  validate `attention_out` only.
- Sequence lengths and `blockshape` are host integer sequences; never pass NPU
  tensors for them. `blockshape[1]` (the KV-direction tile) must be a multiple
  of 128, otherwise tiling fails.
- No extra validation, dtype conversion or contiguity copy is added around the
  ACLNN call; inputs must already satisfy the operator contract.

## torch_native backend

- Pure PyTorch FP32 reference used for accuracy comparison; the result is cast
  back to the query dtype.
- Implements `mask_type=0` (no mask) only and rejects `attn_mask` and
  `block_table`.
- `select_idx` / `select_num_idx` are read as flattened `[q_block, head]` rows.
  GQA query heads map to KV heads in contiguous groups, matching the Kernel's
  `head // (query_heads / kv_heads)`.
- The device Kernel may leave `softmax_lse` untouched, so the reference returns a
  zero tensor of the allocated shape; only `attention_out` is compared.

## Example

```python
import math

import torch
import torch_npu

from sgl_kernel_npu.operators.attention import rain_fusion_attention

batch, heads, seq, dim, block = 1, 2, 256, 128, 128
query = torch.randn(batch * seq, heads, dim, device="npu", dtype=torch.float16)
key = torch.randn(batch * seq, heads, dim, device="npu", dtype=torch.float16)
value = torch.randn(batch * seq, heads, dim, device="npu", dtype=torch.float16)

q_blocks, kv_blocks = seq // block, seq // block
select_idx = torch.arange(kv_blocks, dtype=torch.int64, device="npu")
select_idx = select_idx.repeat(q_blocks, heads, 1)
select_num_idx = torch.full((q_blocks, heads), kv_blocks, dtype=torch.int64, device="npu")

out, _ = rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    [block, block],
    actual_seq_qlen=[seq] * batch,
    actual_seq_kvlen=[seq] * batch,
    head_num=heads,
    scale=dim**-0.5,
)
print(out.shape)  # [256, 2, 128]
```
