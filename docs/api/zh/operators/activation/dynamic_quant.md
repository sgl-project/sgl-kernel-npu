# dynamic_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |


## 函数原型

```python
from sgl_kernel_npu.operators.activation import dynamic_quant

def dynamic_quant(
    x: torch.Tensor,
    dst_type: torch.dtype = torch.int8,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` 与 `torch_npu` 调用 `torch_npu.npu_dynamic_quant`；`torch_native` 使用纯
PyTorch 参考实现。后端按需延迟导入。

## 功能说明

沿最后一维做逐行（per-row）对称 INT8 动态量化：每一行（最后一维）共享一个
scale `amax / 127`，其中 `amax` 为该行绝对值最大值。缩放后的数值四舍五入并截断到
±127，再转为 INT8。整行为 0 时 scale 为 `0`，payload 为 0。

$$
s_r = \frac{\max_{i \in r} |x_i|}{127}, \qquad
y_i = \mathrm{INT8}\big(\mathrm{clamp}(\mathrm{round}(x_i / s_r), \pm 127)\big).
$$

典型用途是 GLM-5.3-Flash DSA indexer 的 INT8 query 量化：`[tokens, heads, 128]`
reshape 为 `[-1, 128]` 后调用，得到 INT8 payload 与每行一个的 FP32 scale，供后续
量化 indexer 计算消费。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]`，`D > 0` | `torch.bfloat16`、`torch.float16`、`torch.float32` | 必填 | NPU 上的连续 Tensor，至少一维。 |
| `dst_type` | `torch.dtype` | — | — | `torch.int8` | 仅支持 `torch.int8`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"torch_npu"`、`"torch_native"`。 |

公开 operator 在分发前校验维度、最后一维宽度、dtype、连续性与 `dst_type`；
违反时抛出 `ValueError`，非连续输入抛出 `RuntimeError`。量化沿最后一维逐行进行，
不对外暴露 axis。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `payload` | `torch.Tensor` | 与 `x` 相同 | `torch.int8` | 与输入同设备的新建连续 Tensor。 |
| `scale` | `torch.Tensor` | `x.shape[:-1]` | `torch.float32` | 每行一个反量化 scale（`amax / 127`）；整行为 0 时 scale 为 `0`。 |

## torch_npu 后端

- 直接绑定厂商算子 `torch_npu.npu_dynamic_quant`，不新增 Ascend C Kernel。
- 输入先 reshape 为 `[-1, D]` 调用厂商算子，再把 payload reshape 回 `x.shape`、scale 规范化为 `x.shape[:-1]` 的 `torch.float32`。
- 量化语义（逐行 `amax / 127` 对称量化、±127 饱和）由厂商算子保证。

## torch_native 后端

- 纯 PyTorch 参考实现位于 `implementation/torch_native/activation/dynamic_quant.py`，不调用 Kernel，也可在 CPU 上运行。
- 复刻逐行 `amax / 127` scale 构造（整行为 0 时 scale 为 `0`）以及带 ±127 饱和的 INT8 转换，支持任意 `[..., D]` 形状与 bf16/fp16/fp32 输入。
- 不模拟 Kernel 的逐位舍入差异，精度比较应基于反量化结果并使用适合 dtype 的容差，而非逐位比较。仅用于精度对拍，不用于性能评估。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_quant

x = torch.randn(4096, 128, dtype=torch.bfloat16, device="npu")
payload, scale = dynamic_quant(x)  # auto -> torch_npu
assert payload.shape == (4096, 128)
assert payload.dtype == torch.int8
assert scale.shape == (4096,)
assert scale.dtype == torch.float32
```

## 限制说明

- 仅支持 `dst_type=torch.int8`；量化沿最后一维逐行进行。
- 最后一维必须为正；输入必须连续。
- `torch_native` 参考实现仅用于精度对拍，不用于性能评估。
