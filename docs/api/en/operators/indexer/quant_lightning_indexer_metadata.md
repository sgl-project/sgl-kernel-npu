# quant_lightning_indexer_metadata

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | Unsupported | Supported |

## Signature

`from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata`

```python
def quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
    impl="auto"
): ...
```

## Behavior

Builds QLI AICPU scheduling and cross-core merge metadata from lengths, mask, compression ratio, and TopK.

Cumulative and used sequence lengths are INT32. Geometry is `num_heads_q=32`, `num_heads_k=1`, `head_dim=128`. Candidate settings must match the main QLI call; regenerate metadata when lengths or static settings change.

## Parameters

T is query rows, B requests, P physical pages and S slots per page.

| Parameter | Default | Type, shape and meaning |
| --- | --- | --- |
| `cu_seqlens_q` | `None` | Optional INT32 [B+1], cumulative query row boundaries. |
| `cu_seqlens_k` | `None` | Optional INT32 [B+1], cumulative TND key boundaries. |
| `seqused_q` | `None` | Optional INT32 [B], valid query rows per request. |
| `seqused_k` | `None` | Optional INT32 [B], valid key tokens per request. |
| `cmp_residual_k` | `None` | Optional INT32 [B], uncompressed residual tokens for causal compression. |
| `batch_size` | `None` | Optional int B for PA_BBND; TND requires None and infers B from query boundaries. |
| `max_seqlen_q` | `-1` | int, query capacity per request; -1 derives it from sequence lengths. |
| `max_seqlen_k` | `-1` | int, key capacity per request; -1 means infer. |
| `num_heads_q` | Required | int, query heads; use 32. |
| `num_heads_k` | Required | int, shared index key heads; use 1. |
| `head_dim` | Required | int, logical head dimension; use 128. |
| `topk` | Required | int, maximum selected tokens per query, in [1,8192]. |
| `mask_mode` | `0` | int: 0 for visible keys, 3 for causal masking. |
| `cmp_ratio` | `1` | int in [1,128], compression ratio used to calculate causal visibility. |
| `layout_q` | `'TND'` | str, use TND (flattened query tokens). |
| `layout_k` | `'TND'` | str, TND or PA_BBND key storage. |
| `candidate_topk_blocks` | `-1` | int, candidate-block output capacity; -1 disables the output. |
| `candidate_block_size` | `-1` | int, use 8 tokens per candidate block when enabled. |
| `impl` | `'auto'` | str, auto and cannbot_dsl select the same AICPU schedule implementation. |

## Returns

| Value | Meaning |
| --- | --- |
| metadata | INT32 [1024], backend-specific schedule. |

## cannbot_dsl

`auto` selects this backend. Requires A5 and a compatible CANNBot-DSL runtime installed in the environment. This operator does not enforce a runtime version. Inputs share a device. First use triggers JIT compilation; warm actual shapes before graph capture.

Metadata encodes a device-specific schedule, with no unique Torch numerical answer. No torch_native alias is provided. The paired compute reference checks output semantics; this entry tests eager/graph parity after length updates.

## Example

Construct a small input and its metadata:

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
lengths = torch.tensor([64], device="npu", dtype=torch.int32)
attrs = dict(topk=512, layout_q="TND", layout_k="PA_BBND")
metadata = quant_lightning_indexer_metadata(
    cu_seqlens_q=cu,
    seqused_k=lengths,
    num_heads_q=32,
    num_heads_k=1,
    head_dim=128,
    max_seqlen_q=1,
    max_seqlen_k=64,
    **attrs
)
```

## Device test

```bash
python -m pytest tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py --npu-device=0 --enable-perf=false -v -s
```
