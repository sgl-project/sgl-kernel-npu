# dynamic_mx_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 不支持 | 支持 |
| `torch_native` | 不支持 | 支持 |

两个实现均注册于 arch35。厂商算子的支持情况还取决于已安装的 `torch_npu`
和 CANN 版本。

## 函数原型

```python
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

def dynamic_mx_quant(*args, impl="auto", **kwargs):
    ...
```

`auto` 与 `torch_npu` 调用 `torch_npu.npu_dynamic_mx_quant`，
`torch_native` 选择 PyTorch 参考实现。注册表按指定实现分发，不自动切换到其他实现。

## 功能说明

`torch_npu` 实现将位置参数和关键字参数透传给
`torch_npu.npu_dynamic_mx_quant`，并原样返回结果。厂商算子沿指定轴分块，
为每个块计算共享 scale。支持的参数、默认值、布局和 dtype 取决于厂商版本，参见
[torch_npu 接口说明](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md)。

KPool query 场景由调用方指定 `axis=-1` 和 `dst_type=torch.float8_e4m3fn`。
在厂商 per-32 分块配置下，每 32 个元素共享一个 E8M0 scale。对于 `[rows, 128]`
输入，厂商返回每行四个 scale 字节，排列为 `[rows, 2, 2]`。调用方将返回的字节
view 为 E8M0，供 indexer 使用。

## 参数与返回值

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `*args` | — | 厂商输入必填 | 传给所选实现的位置参数。 |
| `impl` | `str` | `"auto"` | 仅限关键字，可选 `"auto"`、`"torch_npu"` 或 `"torch_native"`。 |
| `**kwargs` | — | — | 传给所选实现的关键字参数。厂商版本支持时，可传入 `axis`、`dst_type`、`round_mode`、`block_size` 或 `scale_alg` 等。 |

返回所选实现的 `(payload, scales)` 元组。下方 E4M3FN per-32 示例中，payload 的
shape 与输入相同，dtype 为 `torch.float8_e4m3fn`。scale Tensor 的 dtype 为
`torch.uint8`，shape 为 `[rows, ceil(D / 64), 2]`。scale 字节编码 E8M0 数值，
块数为奇数时填充成对。其他参数配置遵循所安装的厂商接口。

## torch_npu 后端

调用 `torch_npu.npu_dynamic_mx_quant(*args, **kwargs)`。封装保留输入参数及返回
Tensor 的 shape 和 dtype。

## torch_native 后端

`implementation/torch_native/activation/dynamic_mx_quant.py` 中的参考实现覆盖末维
per-32 E4M3FN 量化，scale 使用成对排列的 E8M0 字节。该实现接受 `x` 和
`dst_type`，用于精度对拍。与厂商实现比较时传入
`dst_type=torch.float8_e4m3fn`。该参考实现的覆盖范围不包含厂商接口的其他轴、
block size 和量化模式。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
payload, scales = dynamic_mx_quant(x, axis=-1, dst_type=torch.float8_e4m3fn)
assert payload.shape == (64, 128)
assert payload.dtype == torch.float8_e4m3fn
assert scales.shape == (64, 2, 2)
assert scales.dtype == torch.uint8
```

## 限制说明

- 注册表在 arch35 上提供两个实现。
- 厂商实现支持的参数取决于已安装的 `torch_npu` 和 CANN 版本。
- Torch 参考实现覆盖末维 per-32 E4M3FN 量化。
