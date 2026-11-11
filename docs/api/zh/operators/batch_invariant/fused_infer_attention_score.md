# fused_infer_attention_score

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import (
    fused_infer_attention_score_batch_invariant,
)

operator = fused_infer_attention_score_batch_invariant(*, impl="auto")
attention_out, softmax_lse = operator(query, key, value, **kwargs)
```

公开入口**返回 `OpOverloadPacket` 而不直接计算**。保留 packet 是为了在 NPU 图捕获时
继续使用 `.out` 重载与 schema 校验：

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

该算子转发到
`torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant`。
使用前需先安装外部组件，见[安装说明](installation.md)。

## 功能说明

融合推理注意力。online softmax 的归约顺序与本次调用中的 request 数无关，因此每个
request 的输出在单独计算与批量计算下逐位相同。关键字参数与 aclnn fused infer
attention 的 schema 一致。

## 参数与返回值

| 参数 | 类型 | Shape 与约束 |
| --- | --- | --- |
| `query` | `torch.Tensor` | 布局由 `input_layout` 决定。 |
| `key` | `torch.Tensor` | 同布局；分页 KV 需同时提供 `block_table` 与 `block_size`。 |
| `value` | `torch.Tensor` | 与 `key` 同布局。 |
| `num_heads` | `int` | query 头数。 |
| `num_key_value_heads` | `int` | key/value 头数；`0` 表示与 `num_heads` 相同。 |
| `input_layout` | `str` | 张量布局，如 `"BSH"`、`"BNSD"`、`"TND"`，默认 `"BSH"`。 |
| `scale` | `float` | query/key 乘积的缩放系数。 |
| `actual_seq_lengths` | `list[int]` 或 `None` | query 序列长度；TND 布局使用累加和。 |
| `actual_seq_lengths_kv` | `list[int]` 或 `None` | key/value 序列长度。 |
| `block_table` | `torch.Tensor` 或 `None` | 分页 KV 的页表。 |
| `softmax_lse_flag` | `bool` | 是否返回 log-sum-exp，默认 `False`。 |
| `impl` | `str` | `"auto"` 或 `"batch_invariant_ops"`。 |

其余仅关键字参数原样转发给外部算子，参见 aclnn schema。

| 返回值 | 类型 | 描述 |
| --- | --- | --- |
| `attention_out` | `torch.Tensor` | 按 `input_layout` 排布的注意力输出。 |
| `softmax_lse` | `torch.Tensor` 或未定义 tensor | `softmax_lse_flag=True` 时为 log-sum-exp，否则为未定义 tensor。 |

## 图捕获辅助接口

| 接口 | 用途 |
| --- | --- |
| `fused_infer_attention_score_batch_invariant_get_max_workspace` | 按相同参数返回所需 workspace 大小，接受同一套关键字参数。 |
| `fused_infer_attention_score_batch_invariant_infer_output` | 为 `.out` 分配注意力输出与 log-sum-exp 缓冲。 |
| `fused_infer_attention_score_batch_invariant_out` | 语义与默认重载相同，写入调用方提供的缓冲。 |

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

## 示例

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
