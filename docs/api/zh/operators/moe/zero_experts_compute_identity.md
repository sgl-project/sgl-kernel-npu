# zero_experts_compute_identity

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.moe import zero_experts_compute_identity


def zero_experts_compute_identity(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,
    hidden_states,
    identity_mask_value=0,
    *,
    impl="auto"
): ...
```

## 功能描述

计算 LongCat MoE 恒等专家的加权输出。专家 ID >= num_experts 的路由贡献其权重乘 hidden_states，随后这些路由的 scale 清零、ID 替换。若一行所有路由都是恒等专家，第一个替换 ID 为零。两个路由张量均原地修改。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `expert_indices` | Tensor | `Required` | 可写连续整数路由 [tokens,topk]。 |
| `expert_scales` | Tensor | `Required` | 可写连续浮点路由权重 [tokens,topk]。 |
| `num_experts` | int | `Required` | 大于等于该数量的 ID 表示恒等专家。 |
| `zero_expert_type` | str | `Required` | 使用 identity；其他字符串不会选择不同计算。 |
| `hidden_states` | Tensor | `Required` | 连续浮点状态 [tokens,hidden]。 |
| `identity_mask_value` | int | `0` | 恒等专家替换后的路由 ID。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | [tokens,hidden] 加权恒等输出，类型与 hidden_states 相同；修改路由 ID 和 scale。 |

## triton 后端

- 按 token 汇总 identity expert 的路由权重，并将加权 hidden_states 写入新输出。
- 同时原地清零对应权重、替换 expert 索引；整行均为 identity 时保留首个索引为 0。

## torch_native 后端

- 通过 PyTorch 掩码、行归约和乘法计算 identity 分支贡献，输出使用 hidden_states dtype。
- 使用相同 identity 掩码原地更新 expert_indices 和 expert_scales，包括全 identity 行的首索引处理。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import zero_experts_compute_identity

indices = torch.tensor([[0, 16]], device="npu", dtype=torch.int32)
scales = torch.tensor([[0.75, 0.25]], device="npu")
x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
y = zero_experts_compute_identity(indices, scales, 16, "identity", x, impl="triton")
```
