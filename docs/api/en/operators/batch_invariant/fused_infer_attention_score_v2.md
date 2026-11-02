# fused_infer_attention_score_v2

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import (
    fused_infer_attention_score_batch_invariant_v2,
)

attention_out, softmax_lse = fused_infer_attention_score_batch_invariant_v2(
    query, key, value, **kwargs
)
```

The operator forwards to
`torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant_v2`.
Install the external packages first; see [installation](installation.md).

## Description

Fused infer attention v2. The online softmax reduction order does not depend on
how many requests share the call, so each request's output is bitwise identical
whether it runs alone or with others. The keyword-only arguments follow the
aclnn fused infer attention v2 schema.

## Parameters and return value

| Parameter | Type | Shape and constraints |
| --- | --- | --- |
| `query` | `torch.Tensor` | Layout given by `input_layout`. |
| `key` | `torch.Tensor` | Same layout; paged key/value requires `block_table` and `block_size`. |
| `value` | `torch.Tensor` | Same layout as `key`. |
| `num_query_heads` | `int` | Query head count. |
| `num_key_value_heads` | `int` | Key/value head count; `0` means the same as `num_query_heads`. |
| `input_layout` | `str` | Tensor layout such as `"BSH"`, `"BNSD"`, `"TND"`. Defaults to `"BSH"`. |
| `softmax_scale` | `float` | Multiplier applied to the query/key product. |
| `actual_seq_qlen` | `list[int]` or `None` | Query sequence lengths; TND expects cumulative sums. |
| `actual_seq_kvlen` | `list[int]` or `None` | Key/value sequence lengths. |
| `block_table` | `torch.Tensor` or `None` | Page table for paged key/value. |
| `return_softmax_lse` | `bool` | Allocate and return the log-sum-exp result. Defaults to `False`. |
| `impl` | `str` | `"auto"` or `"batch_invariant_ops"`. |

Additional keyword-only arguments (quantization scales, rope tensors, dtype
overrides and so on) are forwarded verbatim; see the aclnn v2 schema.

| Return value | Type | Description |
| --- | --- | --- |
| `attention_out` | `torch.Tensor` | Attention output in `input_layout`. |
| `softmax_lse` | `torch.Tensor` or undefined | Log-sum-exp when `return_softmax_lse` is True; otherwise an undefined tensor. |

## Related helpers

| Helper | Purpose |
| --- | --- |
| `fused_infer_attention_score_batch_invariant_v2_get_max_workspace` | Workspace size for the same arguments. |
| `fused_infer_attention_score_batch_invariant_v2_infer_output` | Allocates the attention output and log-sum-exp buffers. |

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import (
    fused_infer_attention_score_batch_invariant_v2,
)

query = torch.randn(2, 64, 256, device="npu", dtype=torch.bfloat16)
key = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
value = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
kwargs = {
    "num_query_heads": 4,
    "num_key_value_heads": 2,
    "input_layout": "BSH",
    "softmax_scale": 0.125,
    "actual_seq_qlen": [64, 64],
    "actual_seq_kvlen": [64, 64],
}

full, _ = fused_infer_attention_score_batch_invariant_v2(query, key, value, **kwargs)
single, _ = fused_infer_attention_score_batch_invariant_v2(
    query[:1],
    key[:1],
    value[:1],
    **{**kwargs, "actual_seq_qlen": [64], "actual_seq_kvlen": [64]},
)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
