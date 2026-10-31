# fused_infer_attention_score

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import (
    fused_infer_attention_score_batch_invariant,
)

operator = fused_infer_attention_score_batch_invariant(*, impl="auto")
attention_out, softmax_lse = operator(query, key, value, **kwargs)
```

The public entry **returns the `OpOverloadPacket`** instead of calling it. Keep
the packet so the `.out` overload and its schema stay available for NPU graph
capture:

```python
attention_out, softmax_lse = operator.out(
    query,
    key,
    value,
    workspace=workspace,
    out=[attention_out, softmax_lse],
    **kwargs,
)
```

The operator forwards to
`torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant`.
Install the external packages first; see [installation](installation.md).

## Description

Fused infer attention. The online softmax reduction order does not depend on
how many requests share the call, so each request's output is bitwise identical
whether it runs alone or with others. The keyword-only arguments follow the
aclnn fused infer attention schema.

## Parameters and return value

| Parameter | Type | Shape and constraints |
| --- | --- | --- |
| `query` | `torch.Tensor` | Layout given by `input_layout`. |
| `key` | `torch.Tensor` | Same layout; paged key/value requires `block_table` and `block_size`. |
| `value` | `torch.Tensor` | Same layout as `key`. |
| `num_heads` | `int` | Query head count. |
| `num_key_value_heads` | `int` | Key/value head count; `0` means the same as `num_heads`. |
| `input_layout` | `str` | Tensor layout such as `"BSH"`, `"BNSD"`, `"TND"`. Defaults to `"BSH"`. |
| `scale` | `float` | Multiplier applied to the query/key product. |
| `actual_seq_lengths` | `list[int]` or `None` | Query sequence lengths; TND expects cumulative sums. |
| `actual_seq_lengths_kv` | `list[int]` or `None` | Key/value sequence lengths. |
| `block_table` | `torch.Tensor` or `None` | Page table for paged key/value. |
| `softmax_lse_flag` | `bool` | Allocate and return the log-sum-exp result. Defaults to `False`. |
| `impl` | `str` | `"auto"` or `"batch_invariant_ops"`. |

Additional keyword-only arguments are forwarded verbatim to the external
operator; see the aclnn schema.

| Return value | Type | Description |
| --- | --- | --- |
| `attention_out` | `torch.Tensor` | Attention output in `input_layout`. |
| `softmax_lse` | `torch.Tensor` or undefined | Log-sum-exp when `softmax_lse_flag` is True; otherwise an undefined tensor. |

## Graph-capture helpers

| Helper | Purpose |
| --- | --- |
| `fused_infer_attention_score_batch_invariant_get_max_workspace` | Workspace size for the same arguments; accepts the same keyword-only schema. |
| `fused_infer_attention_score_batch_invariant_infer_output` | Allocates the attention output and log-sum-exp buffers for `.out`. |
| `fused_infer_attention_score_batch_invariant_out` | Same semantics as the default overload, writing into caller-provided buffers. |

```python
from sgl_kernel_npu.operators.batch_invariant import (
    fused_infer_attention_score_batch_invariant as fia,
    fused_infer_attention_score_batch_invariant_get_max_workspace as get_workspace,
    fused_infer_attention_score_batch_invariant_infer_output as infer_output,
)

operator = fia()
workspace = get_workspace(query, key, value, **kwargs)
buffers = list(
    infer_output(
        query,
        value,
        input_layout=kwargs["input_layout"],
        num_heads=kwargs["num_heads"],
        num_key_value_heads=kwargs["num_key_value_heads"],
    )
)
operator.out(query, key, value, workspace=workspace, out=buffers, **kwargs)
```

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import (
    fused_infer_attention_score_batch_invariant,
)

query = torch.randn(2, 64, 256, device="npu", dtype=torch.bfloat16)
key = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
value = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
kwargs = {
    "num_heads": 4,
    "num_key_value_heads": 2,
    "input_layout": "BSH",
    "scale": 0.125,
    "actual_seq_lengths": [64, 64],
    "actual_seq_lengths_kv": [64, 64],
}

operator = fused_infer_attention_score_batch_invariant()
full, _ = operator(query, key, value, **kwargs)
single, _ = operator(
    query[:1],
    key[:1],
    value[:1],
    **{**kwargs, "actual_seq_lengths": [64], "actual_seq_lengths_kv": [64]},
)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
