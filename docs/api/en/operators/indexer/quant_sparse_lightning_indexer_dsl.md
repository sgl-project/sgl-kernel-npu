# quant_sparse_lightning_indexer_dsl

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | Unsupported | Supported |
| torch_native | Unsupported | Supported |

## Signature

`from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer_dsl`

```python
import torch
from typing import Optional

def quant_sparse_lightning_indexer_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    descale_k: Optional[torch.Tensor] = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    impl="auto"
): ...
```

## Behavior

Scores only the candidate key blocks selected by QLI and returns sparse TopK tokens.

UINT8 `q[T1,32,64]`, INT32 candidate block indices/lengths, E8M0 key descale, and matching QSLI metadata are required. Use `candidate_block_size=8` and `quant_mode=1`; K may be TND or PA_BBND.

Semantics (BF16 rounding at quantized compute boundaries):

```text
s_i = sum_h w_h * max(dot(dequant(q_h), dequant(k_i)), 0), i in visible candidate blocks; indices = TopK(s)
```

`dequant(code, e) = signed_E2M1(code) * 2**(e - 127)`, with one E8M0 exponent byte per 32 logical values and no zero point. E2M1 uses magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`; the first value is in the low nibble. QK results, head weights, and each weighted head accumulation are rounded to BF16.

## Parameters

T is query rows, B requests, P physical pages and S slots per page.

| Parameter | Default | Type, shape and meaning |
| --- | --- | --- |
| `q` | Required | UINT8 [T,32,64], packed MXFP4 queries with logical dimension 128. |
| `k` | Required | UINT8 K: TND [K,1,64] or PA_BBND [P,S,1,64]. QSLI PA uses combined [P,S/8,544] bytes. |
| `w` | Required | FP32 [T,32], query-head score weights. |
| `descale_q` | Required | UINT8/E8M0 [T,32,2,2], one scale per 32 logical features. |
| `candidate_block_indices` | Required | INT32 [T,1,2048], candidate logical block IDs; valid prefix given by lengths. |
| `candidate_block_length` | Required | INT32 [T,1], valid candidate block count per query. |
| `cu_seqlens_q` | `None` | Optional INT32 [B+1], cumulative query row boundaries. |
| `cu_seqlens_k` | `None` | Optional INT32 [B+1], cumulative TND key boundaries. |
| `seqused_q` | `None` | Optional INT32 [B], valid query rows per request. |
| `seqused_k` | `None` | Optional INT32 [B], valid key tokens per request. |
| `cmp_residual_k` | `None` | Optional INT32 [B], uncompressed residual tokens for causal compression. |
| `block_table` | `None` | Optional INT32 [B,max_blocks], logical-to-physical PA page map. |
| `output_idx_offset` | `None` | Optional INT32 [T,1], added to each valid selected token index. |
| `metadata` | `None` | INT32 [1024] device schedule required by cannbot_dsl; torch_native ignores it. |
| `descale_k` | `None` | UINT8/E8M0 [K,1,2,2] or [P,S,1,2,2]; omit for QSLI combined PA storage. |
| `topk` | Required | int, maximum selected tokens per query, in [1,8192]. |
| `candidate_block_size` | Required | int, use 8 tokens per candidate block when enabled. |
| `quant_mode` | Required | int, use 1 for the supported quantized profile. |
| `max_seqlen_q` | `-1` | int or None, query capacity per request; infer when unspecified. |
| `mask_mode` | `0` | int: 0 for visible keys, 3 for causal masking. |
| `cmp_ratio` | `1` | int in [1,128], compression ratio used to calculate causal visibility. |
| `layout_q` | `'TND'` | str, use TND (flattened query tokens). |
| `layout_k` | `'TND'` | str, TND or PA_BBND key storage. |
| `return_value` | `False` | bool, also return BF16 selected scores when True. |
| `impl` | `'auto'` | str, auto selects cannbot_dsl; torch_native is an independent compute reference. |

## Returns

| Value | Meaning |
| --- | --- |
| indices | INT32 [T,1,topk]; -1 marks invalid entries. |
| values | BF16 [T,1,topk]; empty when disabled; invalid entries are -inf. |

## cannbot_dsl

`auto` selects this backend. Requires A5 and a compatible CANNBot-DSL runtime installed in the environment. This operator does not enforce a runtime version. Inputs share a device. First use triggers JIT compilation; warm actual shapes before graph capture.

## torch_native

Independent CPU Torch accuracy reference; results are copied back to the input device, preserving in-place semantics. Does not require DSL. Not intended for performance or NPU graph capture. Equal-score TopK order may differ.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import (
    quant_sparse_lightning_indexer_dsl,
    quant_sparse_lightning_indexer_metadata_dsl,
)

q = torch.zeros(1, 32, 64, device="npu", dtype=torch.uint8)
qs = torch.full((1, 32, 2, 2), 127, device="npu", dtype=torch.uint8)
w = torch.ones(1, 32, device="npu")
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
lengths = torch.tensor([64], device="npu", dtype=torch.int32)
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
attrs = dict(topk=512, layout_q="TND", layout_k="PA_BBND")
k = torch.zeros(1, 8, 544, device="npu", dtype=torch.uint8)
k[..., 512:] = 127
candidates = torch.full((1, 1, 2048), -1, device="npu", dtype=torch.int32)
candidates[..., :8] = torch.arange(8, device="npu", dtype=torch.int32)
count = torch.tensor([[8]], device="npu", dtype=torch.int32)
metadata = quant_sparse_lightning_indexer_metadata_dsl(
    cu_seqlens_q=cu,
    seqused_k=lengths,
    num_heads_q=32,
    num_heads_k=1,
    head_dim=128,
    max_seqlen_q=1,
    max_seqlen_k=64,
    **attrs,
    candidate_block_length=count,
    candidate_block_size=8,
    quant_mode=1
)
result = quant_sparse_lightning_indexer_dsl(
    q,
    k,
    w,
    qs,
    candidates,
    count,
    cu_seqlens_q=cu,
    seqused_k=lengths,
    block_table=table,
    metadata=metadata,
    quant_mode=1,
    **attrs,
    candidate_block_size=8
)
```

## Device test

```bash
python -m pytest tests/python/operators/indexer/test_quant_sparse_lightning_indexer_dsl.py --npu-device=0 --enable-perf=false -v -s
```
