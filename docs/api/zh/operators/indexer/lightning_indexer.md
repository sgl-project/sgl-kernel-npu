# lightning_indexer

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.indexer import lightning_indexer


def lightning_indexer(query, key, weights, *, impl="auto", **kwargs): ...
```

## 功能说明

将 query 与候选 key 的点积做 ReLU，再用头权重汇总并排序。设头权重为 $w_h$，分数为 $\sum_h w_h\max(0,q_h\cdot k_j)$，由 SDK 选取指定数量的位置。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [B, Sq, H, D] for BSND 或 [T, H, D] for TND, FP16/BF16 | 必传 | query 头。分页 DSA 场景中 D 为 128。 |
| `key` | Tensor | [P, L, Hkv, D]，用于 PA_BSND, query 的 dtype | 必传 | 头宽匹配的键缓存。SDK 也提供稠密布局选项。 |
| `weights` | Tensor | [B, Sq, H] 或 [T, H], query 的 dtype as supported | 必传 | 每个 query 头一个分数系数。 |
| `actual_seq_lengths_query`, `actual_seq_lengths_key` | Tensor or None | [B], INT32 | `None` | 与 query 和 key 布局匹配的可选 SDK 长度参数。 |
| `block_table` | Tensor or None | [B, max_pages], INT32 | `None` | 可选逻辑页到物理页映射，分页示例中必传。 |
| `layout_query`, `layout_key` | str | - | `"BSND"` | 可选布局标签，默认均为 BSND。示例中 key 为 PA_BSND。 |
| `sparse_count`, `sparse_mode`, `return_value` | int, int, bool | - | `2048` / `3` / `False` | 可选 SDK 参数，默认分别为 2048、3 和 False。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `indices`, `values` | tuple[Tensor, Tensor] | [B, Sq, Hkv, K] for BSND 或 [T, Hkv, K] for TND | INT32 top-k 位置索引。return_value=False 时分数输出为空。 |

## torch_npu 后端

调用 `torch_npu.npu_lightning_indexer`。形状、dtype 和布局组合以已安装的 SDK 为准。 当前 OpPlugin 产品文档列出了 950DT。请按所用 SoC 和布局核对已安装的 TorchNPU。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.indexer import lightning_indexer

q = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16, device="npu")
key = torch.randn((32, 256, 1, 128), dtype=q.dtype, device=q.device)
weights = torch.ones((1, 1, 64), dtype=q.dtype, device=q.device)
q_lens = torch.tensor([1], dtype=torch.int32, device=q.device)
k_lens = torch.tensor([8192], dtype=torch.int32, device=q.device)
table = torch.arange(32, dtype=torch.int32, device=q.device).reshape(1, 32)
indices, values = lightning_indexer(
    q,
    key,
    weights,
    actual_seq_lengths_query=q_lens,
    actual_seq_lengths_key=k_lens,
    block_table=table,
    layout_key="PA_BSND",
    sparse_count=2048,
    impl="torch_npu",
)
```
