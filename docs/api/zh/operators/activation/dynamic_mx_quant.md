# dynamic_mx_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 不支持 | 支持 |
| `torch_native` | 不支持 | 支持 |


## 函数原型

```python
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

def dynamic_mx_quant(
    x: torch.Tensor,
    dst_type: torch.dtype = torch.float8_e4m3fn,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` 与 `torch_npu` 调用 `torch_npu.npu_dynamic_mx_quant`（固定 `axis=-1`，
block size 为 32）；`torch_native` 使用纯 PyTorch 参考实现。后端按需延迟导入。
厂商算子直接调用，不回退；需要 PyTorch 参考时显式传入 `impl="torch_native"`。

## 功能说明

沿最后一维做动态 MXFP8 量化：每 32 个连续元素共享一个 E8M0（2 的整数次幂）
scale，由该组最大绝对值的 floor(log2) 指数推导，并保证缩放后的最大值落在
E4M3FN 范围内。缩放后的数值先显式截断到 ±448，再以 round-to-nearest-even 转为
E4M3FN；该 ±448 饱和是显式行为，而不只是格式有限最大值带来的隐含限制。整组为 0
时 scale 字节为 `0`；组内出现非有限值时 scale 字节为 `255`，对应 payload 为 NaN。

$$
e_g = \lfloor \log_2 \max_{i \in g} |x_i| \rfloor, \qquad
s_g = 2^{e_g - 8}, \qquad
y_i = \mathrm{E4M3FN}\big(\mathrm{clamp}(x_i / s_g, \pm 448)\big).
$$

E8M0 scale 字节为 `e_g - 8 + 127`（下限截断到 `0`）。

典型用途是 GLM-5.3-Flash DSA indexer 的 query 量化：`[tokens, heads, 128]`
reshape 为 `[-1, 128]` 后调用，得到 E4M3FN payload 与每 32 元素一个的 E8M0
scale，供后续量化 indexer 计算消费。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]`，`D > 0` 且 `D % 32 == 0` | `torch.bfloat16`、`torch.float16`、`torch.float32` | 必填 | NPU 上的连续 Tensor，至少一维。 |
| `dst_type` | `torch.dtype` | — | — | `torch.float8_e4m3fn` | 仅支持 `torch.float8_e4m3fn`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"torch_npu"`、`"torch_native"`。 |

公开 operator 在分发前校验维度、最后一维宽度、dtype、连续性与 `dst_type`；
违反时抛出 `ValueError`，非连续输入抛出 `RuntimeError`。`axis=-1` 与 block
size 32 固定，不对外暴露。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `payload` | `torch.Tensor` | 与 `x` 相同 | `torch.float8_e4m3fn` | 与输入同设备的新建连续 Tensor。 |
| `scales` | `torch.Tensor` | `x.shape[:-1] + (D // 32,)` | `torch.float8_e8m0fnu` | 每 32 元素一个 E8M0 scale 字节；厂商返回的 uint8 或 `[..., D // 64, 2]` 布局会被规范化为该形状。 |

## torch_npu 后端

- 直接绑定厂商算子 `torch_npu.npu_dynamic_mx_quant`，固定 `axis=-1`，block size 使用厂商默认的 32；不新增 Ascend C Kernel。
- 量化语义（E8M0 scale 推导、±448 饱和、`0`/`255` 特殊字节）由厂商 micro-scaling 原语保证。
- 厂商 scale 输出统一规范化为 `[..., D // 32]` 的 `torch.float8_e8m0fnu`。
- 不静默回退到其他实现；arch22 上由 generation 注册表报告为不可用。

## torch_native 后端

- 纯 PyTorch 参考实现位于 `implementation/torch_native/activation/dynamic_mx_quant.py`，不调用 Kernel，也可在 CPU 上运行。
- 复刻按 32 元素分组的 E8M0 scale 构造（整组为 0 时为 `0`，非有限值时为 `255`）以及带 ±448 饱和的 E4M3FN 转换，支持任意 `[..., D]` 形状与 bf16/fp16/fp32 输入。
- 不模拟 Kernel 的逐位舍入差异，精度比较应基于反量化结果并使用适合 dtype 的容差，而非逐位比较。仅用于精度对拍，不用于性能评估。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

x = torch.randn(4096, 128, dtype=torch.bfloat16, device="npu")
payload, scales = dynamic_mx_quant(x)  # auto -> torch_npu
assert payload.shape == (4096, 128)
assert payload.dtype == torch.float8_e4m3fn
assert scales.shape == (4096, 4)
assert scales.dtype == torch.float8_e8m0fnu
```

## 限制说明

- 仅支持 A5；arch22 上所有实现均不可用。
- 仅支持 `dst_type=torch.float8_e4m3fn`；`axis=-1` 与 block size 32 固定。
- 最后一维必须为正的 32 的倍数；输入必须连续。
- `torch_native` 参考实现仅用于精度对拍，不用于性能评估。
