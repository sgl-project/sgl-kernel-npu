# split_qkv_tp_rmsnorm_rope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `cpu` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

q, k, v = split_qkv_tp_rmsnorm_rope(
    input, cos, sin, q_hidden_size, kv_hidden_size, head_dim, eps,
    q_weight, k_weight, rotary_dim, tp_world, tp_group, impl="auto",
)
```

## Parameters and return values

`input` is the dense rank-local Q, K, V concatenation of shape `[B, q_hidden_size + 2 * kv_hidden_size]`. Both widths are local shard widths. Each row has separate Q and K mean squares over the respective local width. Use equal batch sizes and equal shard widths across ranks. Set `tp_world` to the size of `tp_group`. Use `1, None` on a single device.

`head_dim` is a power of two. Local widths contain whole heads and the Q width is a multiple of the KV width. `q_weight` and `k_weight` contain the local channel weights. Add `eps` to the global mean square before taking the square root. Each row of `cos` and `sin` contains `rotary_dim` elements. The first half rotates the first `rotary_dim` channels of each head in NeoX order.

Returns separate Q, K and V tensors of shapes `[B, q_hidden_size]`, `[B, kv_hidden_size]` and `[B, kv_hidden_size]` on the input device and in the input dtype. V is copied. Q and K use FP32 statistics and round the weighted normalized values to input precision before RoPE.

## Implementations

`auto` selects Triton. When `tp_world > 1`, a SUM all-reduce combines per-row Q/K mean squares between the two compute stages, followed by division by the TP size. `torch_native` computes on the input device. `cpu` computes reference values on CPU, uses the input device for the process-group reduction and returns outputs on the input device.
