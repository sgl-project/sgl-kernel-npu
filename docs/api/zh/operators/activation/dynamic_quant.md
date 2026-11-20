# dynamic_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

注册表在两个代际提供以上实现。厂商算子在各设备上支持的参数取决于已安装的
`torch_npu` 和 CANN 版本。

## 函数原型

```python
from sgl_kernel_npu.operators.activation import dynamic_quant

def dynamic_quant(*args, impl="auto", **kwargs):
    ...
```

`auto` 与 `torch_npu` 调用 `torch_npu.npu_dynamic_quant`，
`torch_native` 选择 PyTorch 参考实现。注册表按指定实现分发，不自动切换到其他实现。

## 功能说明

`torch_npu` 实现将位置参数和关键字参数透传给
`torch_npu.npu_dynamic_quant`，并原样返回结果。厂商算子支持 per-token 等动态量化模式。
支持的参数、dtype、shape 和量化模式取决于厂商版本及设备，参见
[torch_npu 接口说明](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)。

KPool indexer query 场景先由调用方进行 Hadamard 旋转，将 query 整理为
`[rows, 128]` 并指定 `dst_type=torch.int8`。在这一逐行对称量化场景中，scale
由每行最大绝对值和 INT8 范围计算。调用方再恢复 query shape，并将 scale 转为 indexer
需要的 dtype。

## 参数与返回值

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `*args` | — | 厂商输入必填 | 传给所选实现的位置参数。 |
| `impl` | `str` | `"auto"` | 仅限关键字，可选 `"auto"`、`"torch_npu"` 或 `"torch_native"`。 |
| `**kwargs` | — | — | 传给所选实现的关键字参数。厂商版本支持时，可传入 `dst_type`、`smooth_scales`、`group_index` 或 `quant_mode` 等。 |

返回所选实现的 `(payload, scale)` 元组。下方 INT8 per-token 示例中，payload 的
shape 与输入相同，dtype 为 `torch.int8`，scale 的 shape 为 `[rows]`，dtype 为
`torch.float32`。其他模式遵循所安装的厂商接口。

## torch_npu 后端

调用 `torch_npu.npu_dynamic_quant(*args, **kwargs)`。封装保留输入参数及返回
Tensor 的 shape 和 dtype。

## torch_native 后端

`implementation/torch_native/activation/dynamic_quant.py` 中的参考实现沿最后一维
进行逐行对称 INT8 量化，返回 INT8 payload 和每行一个 FP32 scale。全零行的 scale 为
0。该实现接受 `x` 和 `dst_type`，用于精度对拍，其覆盖范围不包含 smoothing
和其他厂商量化模式。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_quant

x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
payload, scale = dynamic_quant(x, dst_type=torch.int8)
assert payload.shape == (64, 128)
assert payload.dtype == torch.int8
assert scale.shape == (64,)
assert scale.dtype == torch.float32
```

## 限制说明

- 厂商实现支持的输入和参数取决于已安装的 `torch_npu`、CANN 和设备版本。
- Torch 参考实现覆盖逐行对称 INT8 量化。
