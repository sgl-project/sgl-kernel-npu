# mrope

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.rope import mrope


def mrope(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode="half",
    cache_mode="default",
    impl="auto"
): ...
```

## 功能说明

按位置查找旋转系数，并旋转每个 query 和 key 头的 RoPE 前缀。多模态位置先为各轴分段选择缓存行，再执行 RoPE，公式为 $x\odot\cos+\mathrm{rotate}(x)\odot\sin$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `positions` | Tensor | [T] 或 [axes, T], INT32/INT64 | 必传 | 缓存范围内的单轴或多轴位置索引。 |
| `query`, `key` | Tensor | [T, H * head_size], FP16/BF16 | 必传 | 打包后的 query 和 key，各自的头数可不同。 |
| `cos_sin_cache` | Tensor | [max_seq_len, rotary_dim], floating | 必传 | 三角函数缓存。默认布局中每行先存余弦频率，再存正弦频率。 |
| `head_size` | int | - | 必传 | 每个打包的 query 或 key 头的宽度。 |
| `mrope_section` | list[int] or None | - | `None` | 将一半旋转频率划分到各多模态轴。 |
| `rotary_mode` | str | - | `'half'` | 旋转坐标配对方式，默认为 half。 |
| `cache_mode` | str | - | `'default'` | 三角函数缓存布局。其他布局由已安装 SDK 决定。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `query_out`, `key_out` | tuple[Tensor, Tensor] | corresponding input shape 与 dtype | 输入设备上旋转后的 query 和 key。 |

## torch_npu 后端

调用 `torch_npu.npu_mrope`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mrope.md)

## torch_native 后端

支持 cache_mode='default' 下的分段查表，并提供半区和相邻配对。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import mrope

pos = torch.arange(7, dtype=torch.int64, device="npu")
q = torch.randn((7, 4 * 128), dtype=torch.float16, device="npu")
k = torch.randn((7, 128), dtype=q.dtype, device=q.device)
theta = torch.zeros((9, 64), dtype=q.dtype, device=q.device)
cache = torch.cat((theta.cos(), theta.sin()), dim=-1)
q_out, k_out = mrope(pos, q, k, cache, 128, impl="torch_npu")
```
