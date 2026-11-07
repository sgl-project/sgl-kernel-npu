# situ_mxfp8_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 不支持 | 支持 |

该算子仅支持 A5：`SUPPORTED_ARCHS` 为 `3510`，arch22 不会编译该算子，Python 层在
arch22 上会把所有实现（包括 PyTorch 参考实现）报告为不可用。arch35 已在 Ascend 950
上验证。

## 函数原型

```python
from sgl_kernel_npu.operators.activation import situ_mxfp8_quant

def situ_mxfp8_quant(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`auto` 与 `ascendc` 调用 `torch.ops.npu.situ_mxfp8_quant`；`torch_native` 使用纯
PyTorch 参考实现。后端按需延迟导入。arch22 wheel 或未包含 `situ_mxfp8_quant` 的子集
wheel 不会注册原生算子，且不回退：除非显式传入 `impl="torch_native"`，Torch 查找会
直接失败。

## 功能说明

将 Kimi-K3 的 SiTU 激活与按 32 元素分组的 MXFP8 量化融合。输入是 GMM1 的 BF16 输出
`[capacity, 6144]`，第 `[0, 3072)` 列为 gate 分支，第 `[3072, 6144)` 列为 up 分支。
[SGLang
NPUSituMXFP8Quant](https://github.com/sgl-project/sglang/blob/5481134403e01b46891e091ec63db42a36e6626f/python/sglang/srt/hardware_backend/npu/moe/activation.py#L163-L185)
在打包的 gate/up 投影与分组 down 投影之间调用该算子。

$$
g = x[:, :3072], \qquad u = x[:, 3072:],
$$

$$
a = \beta\tanh(g / \beta)\,\sigma(g), \qquad
v = L\tanh(u / L), \qquad
y = a v, \qquad
L = \mathrm{linear\_beta}.
$$

所有中间计算使用 float32；`y` 按未融合路径的语义以 round-to-nearest-even 舍入到
BF16，然后再做 MXFP8 量化。

3072 列 BF16 结果中每 32 个连续元素共享一个 E8M0（2 的整数次幂）scale，由该组的
最大 BF16 指数推导。缩放后的数值先截断到 ±448，再以 round-to-nearest-even 转为
E4M3FN；该 ±448 饱和是显式行为，而不只是格式有限最大值带来的隐含限制。整组为 0 时
scale 字节为 `0`；组内出现非有限值时 scale 字节为 `255`。

`group_list` 只用于确定有效行数：`group_list_type=1` 时为 `sum(group_list)`，
`group_list_type=0` 时为 `group_list[-1]`。算子不消费各 expert 的边界，因此只要总行数
相同，不同的列表分布结果完全一致。仅写前 `min(valid_rows, capacity)` 行，其余行为
未初始化 padding，与下游 grouped matmul 的消费约定一致。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `hidden_states` | `torch.Tensor` | `[capacity, 6144]` | `torch.bfloat16` | 必填 | NPU 上的连续 Tensor，`capacity > 0`；前半为 gate，后半为 up。 |
| `group_list` | `torch.Tensor` | `[E]` | `torch.int32`、`torch.int64` | 必填 | 与输入同一 NPU 上的连续、非空一维 Tensor。 |
| `group_list_type` | `int` | — | — | `1` | `0` 为每 expert 累积计数（最后一个元素是有效行数），`1` 为每 expert 计数（求和为有效行数）。仅使用该行数。 |
| `beta` | `float` | — | — | `4.0` | gate 分支的 soft-saturation 上界，必须为正。 |
| `linear_beta` | `float` | — | — | `25.0` | up 分支的 soft-saturation 上界，必须为正。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"ascendc"`、`"torch_native"`。 |

公开 operator 在分发前校验固定宽度、dtype、连续性、`group_list` 的
shape/dtype/device、`group_list_type` 以及正数界限；违反时抛出 `ValueError`，
非连续输入抛出 `RuntimeError`。设备端的 group 数值由调用方保证：算子不做校验，
总和超过 `capacity` 时按 `capacity` 截断。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `payload` | `torch.Tensor` | `[capacity, 3072]` | `torch.float8_e4m3fn` | 与输入同设备的新建连续 Tensor；仅有效前缀有定义。 |
| `scales` | `torch.Tensor` | `[capacity, 48, 2]` | `torch.float8_e8m0fnu` | 与厂商 `npu_dynamic_mx_quant` 布局一致：每行 96 个连续 E8M0 字节（每 32 元素一个），按 `[48, 2]` 视图返回。 |

## ascendc 后端

- 单一 CANN Kernel 入口 `sgl_situ_mxfp8_quant`，构建声明为 `SUPPORTED_ARCHS 3510`。tiling 沿用旧实现的 `blockDim = min(GetCoreNumAiv(), rows)` 调度，每个核处理 `ceil_div(valid_rows, blockDim)` 行。
- 问题规模固定为 `[capacity, 6144] -> [capacity, 3072]`，不支持动态 shape 或列方向分块。
- Kernel 在设备端读取 `group_list`，归约为有效行数后只处理该前缀，不使用各 expert 的分区边界。
- SiTU 使用 fp32 计算，舍入到 BF16 后通过 `mx_quant_kernel.h` 中的厂商 micro-scaling 原语量化；E4M3FN 使用 round-to-nearest-even 并显式截断到 ±448。
- 不写 padding 行，其内容未定义，不得读取。

## torch_native 后端

- 纯 PyTorch 参考实现位于 `implementation/torch_native/activation/situ_mxfp8_quant.py`，不调用 Kernel，也可在 CPU 上运行。
- 复刻 fp32 SiTU、BF16 舍入、按 32 元素分组的 E8M0 scale 构造（整组为 0 时为 `0`，非有限值时为 `255`）以及带 ±448 饱和的 E4M3FN 转换。
- 不模拟 Kernel 的逐位舍入差异，精度比较应基于反量化结果并使用适合 dtype 的容差，而非逐位比较。仅用于精度对拍，不用于性能评估。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import situ_mxfp8_quant

capacity = 128
counts = torch.tensor([0, 7, 13, 0, 9, 3, 0, 5], dtype=torch.int64, device="npu")
x = (torch.randn(capacity, 6144, dtype=torch.float32) * 2).to(torch.bfloat16).npu()

payload, scales = situ_mxfp8_quant(x, counts, group_list_type=1)  # auto -> ascendc
assert payload.shape == (capacity, 3072)
assert scales.shape == (capacity, 48, 2)
```

## 限制说明

- 问题规模固定为 `[capacity, 6144] -> [capacity, 3072]`。
- 仅支持 A5；arch22 wheel 不包含该算子，Python 层在该代际拒绝所有实现。
- 设备端 `group_list` 数值不做校验；总和超过 `capacity` 时按 `capacity` 截断，不报错。
- padding 行按设计未定义。
- `torch_native` 参考实现仅用于精度对拍，不用于性能评估。
