# interleave_rope

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-transformer`。

```python
from sgl_kernel_npu.operators.rope import interleave_rope


def interleave_rope(x, cos, sin, *, impl="auto"): ...
```

## 功能说明

将偶数和奇数坐标分别排入两个半区，得到 $z$，再按重排后的顺序计算 $y=z\odot\cos+\mathrm{rotateHalf}(z)\odot\sin$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [T, H, S, D], FP16/BF16/FP32 | 必传 | ND 输入，D 为偶数。 |
| `cos`, `sin` | Tensor | broadcastable to x, x.dtype | 必传 | 按解交错顺序排列的可广播旋转系数。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | same as x | 按解交错顺序排列的旋转结果。 |

## ops-transformer 后端

调用 `torch_npu.npu_interleave_rope`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_interleave_rope.md) [CANN ops-transformer 源码](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/posembedding/interleave_rope)

## torch_native 后端

用 PyTorch 拼接偶数与奇数列，再按半区配对计算 RoPE。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import interleave_rope

x = torch.randn((16, 16, 1, 64), dtype=torch.float16, device="npu")
cos = torch.ones((16, 1, 1, 64), dtype=x.dtype, device=x.device)
sin = torch.zeros_like(cos)
y = interleave_rope(x, cos, sin, impl="ops-transformer")
```
