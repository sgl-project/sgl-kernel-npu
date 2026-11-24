# rotary_mul

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-transformer`。

```python
from sgl_kernel_npu.operators.rope import rotary_mul


def rotary_mul(input, r1, r2, rotary_mode="half", rotate=None, *, impl="auto"): ...
```

## 功能说明

计算 $y=x\odot r_1+\mathrm{rotate}(x)\odot r_2$。half 模式将两个半区的对应坐标配对，interleave 模式将相邻坐标配对。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [B, S, H, D], FP16/BF16/FP32 | 必传 | ND 输入，旋转宽度 D 为偶数。 |
| `r1`, `r2` | Tensor | broadcastable to input | 必传 | 按指定布局广播的余弦和正弦因子。 |
| `rotary_mode` | str | - | `'half'` | 配对模式，例如 half 或 interleave。 |
| `rotate` | Tensor or None | SDK-specific | `None` | 已安装 TorchNPU 支持的可选旋转矩阵。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | 与输入一致 | 旋转后的激活。 |

## ops-transformer 后端

调用 `torch_npu.npu_rotary_mul`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md) [CANN ops-transformer 源码](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/posembedding/rotary_position_embedding)

## torch_native 后端

以 FP32 实现半区和相邻配对旋转。要求 rotate=None。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import rotary_mul

x = torch.randn((2, 17, 4, 64), dtype=torch.float16, device="npu")
r1 = torch.ones((1, 17, 1, 64), dtype=x.dtype, device=x.device)
r2 = torch.zeros_like(r1)
y = rotary_mul(x, r1, r2, impl="ops-transformer")
```
