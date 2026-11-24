# la

LaserAttention dense attention operator.

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | FP16 | — |
| `torch_native` | FP16 / BF16 | FP16 / BF16 |

## Signature

```python
from sgl_kernel_npu.operators.attention import la

def la(
    query,
    key,
    value,
    *,
    atten_mask=None,
    alibi_mask=None,
    drop_mask=None,
    scale_value=1.0,
    head_num=2,
    input_layout="BNSD",
    keep_prob=1.0,
    pre_tokens=2147483647,
    next_tokens=1,
    is_highPrecision=True,
    impl="auto",
):
    ...
```

`auto` selects `ascendc` on arch22; `torch_native` uses PyTorch operations.
Requesting `ascendc` on arch35 raises instead of switching backends.

## Description

Computes dense (non-causal) attention over BNSD tensors. Query, key and value
are converted to FP32 for the score/softmax/value products, and the Kernel
emits FP32 outputs:

```text
scores = scale_value * (query @ key^T)
probs  = softmax(scores)
out    = probs @ value
```

Grouped-query attention is supported: `key`/`value` may have fewer heads than
`query`, and each KV head serves `query_heads / kv_heads` query heads.

## Parameters and return values

`B` is the batch size, `Nq`/`Nkv` the query/key head counts, `S` the sequence
length and `D` the head width.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `query` | `[B, Nq, S, D]` | FP16 / BF16 | Must be 4D. |
| `key` | `[B, Nkv, S, D]` | Same as `query` | `Nq % Nkv == 0`. |
| `value` | `[B, Nkv, S, D]` | Same as `query` | Value tensor. |
| `atten_mask`, `alibi_mask` | — | `None` | Accepted for signature compatibility; the dense Kernel ignores them. |
| `drop_mask` | — | `None` | Accepted for signature compatibility; dropout is disabled. |
| `scale_value` | — | float, 1.0 | Score multiplier. |
| `head_num` | — | int, 2 | Query head count; must equal `Nq`. |
| `input_layout` | — | str, `"BNSD"` | Only BNSD is supported. |
| `keep_prob` | — | float, 1.0 | Must stay 1.0. |
| `pre_tokens` | — | int, 2147483647 | Window bound; use the default for the dense path. |
| `next_tokens` | — | int, 1 | Window bound. |
| `is_highPrecision` | — | bool, `True` | High-precision accumulation. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `softmax_log_max_sum` | `[B, Nq, S]` | FP32 |
| `attention_out` | `[B, Nq, S, D]` | FP32 |

## ascendc backend

- Requires all tensors on the same NPU; noncontiguous inputs are converted to
  contiguous working tensors internally.
- `softmax_log_max_sum` is allocated for signature compatibility but the current
  Kernel never writes it, so its content is undefined. Validate only
  `attention_out`.
- The Kernel addresses attention in fixed 128-wide base blocks (`S1/128`,
  `S2/128`) and splits cores two block rows at a time (`ky=2`), so `S` must be
  a multiple of 256 and `D` must be 128; other shapes are not rejected but
  compute wrong results, and `S < 128` deadlocks the AIC/AIV handshake
  (inherited from the legacy operator).
- The Kernel instantiates the FP16 path only: the op declaration lists BF16, but
  the tilingKey=1 the Host selects for BF16 has no Kernel entry, so BF16 calls
  fail at Kernel launch with a funcEntry error (inherited from the legacy
  operator; this migration keeps the original behavior and adds no BF16
  support).

## torch_native backend

- Supports the dense path only: `input_layout="BNSD"`, no masks and
  `keep_prob=1.0`. It raises otherwise, and it ignores `pre_tokens` and
  `next_tokens`.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import la

batch, heads, seq, dim = 1, 2, 256, 128
query = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
key = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
value = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)

softmax_log_max_sum, out = la(
    query, key, value,
    scale_value=dim**-0.5,
    head_num=heads,
)
print(out.shape)  # [1, 2, 256, 128]
```
