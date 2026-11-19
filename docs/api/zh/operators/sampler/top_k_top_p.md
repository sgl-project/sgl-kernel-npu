# top_k_top_p

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.sampler import top_k_top_p


def top_k_top_p(logits, p=None, k=None, *, impl="auto"): ...
```

## 功能说明

屏蔽低于第 k 大的分数，再对 top-k 过滤后的 logits 做 softmax，并按累积概率剔除低概率项。与第 k 大分数并列的项可保留，每行至少保留最高分。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `logits` | Tensor | [B, V], FP16/BF16/FP32 | 必传 | ND 格式的未归一化 token 分数。 |
| `p` | Tensor or None | [B], logits.dtype | `None` | [0, 1] 范围内的 top-p 阈值，p 和 k 至少提供一项。 |
| `k` | Tensor or None | [B], INT32 | `None` | 应用 top-p 前保留的最高分数量，取值为 [1, min(1024, V)]。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | 与 logits 一致 | 词表顺序保持不变，剔除的位置为 -inf。 |

## torch_npu 后端

调用 `torch_npu.npu_top_k_top_p`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_top_k_top_p.md)

## torch_native 后端

将 logits 排序，执行两项过滤，再散射回原词表索引。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sampler import top_k_top_p

x = torch.randn((1, 8192), dtype=torch.float32, device="npu")
p = torch.tensor([0.85], dtype=x.dtype, device=x.device)
k = torch.tensor([32], dtype=torch.int32, device=x.device)
filtered = top_k_top_p(x, p, k, impl="torch_npu")
```
