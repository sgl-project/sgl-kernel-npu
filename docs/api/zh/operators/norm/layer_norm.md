# layer_norm

LayerNorm 归一化算子。

## 硬件支持

| 实现 | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `cann_ops_transformer` | 支持 | 不支持 |
| `torch_native` | 支持 | 不支持 |

## 签名

```python
from sgl_kernel_npu.operators.norm import layer_norm

def layer_norm(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
    impl_mode=0,
    impl="auto",
):
    ...
```

`auto` 在 arch22 上选择 `cann_ops_transformer`。没有回退路径：在 arch35 上请求该算子会直接报错。
同时导出旧名字 `layernorm` 作为别名。

## 功能说明

对 `input` 末尾 `len(normalized_shape)` 个维度做归一化：

```text
out = (input - E[input]) / sqrt(Var[input] + eps) * weight + bias
```

未归一化的前置维度视为独立的行；`weight` / `bias` 为可选仿射参数。

## 参数与返回值

`A...` 表示不参与归一化的前置维度，`R...` 表示参与归一化的维度。

| 参数 | 形状 | 数据类型 / 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | `[A..., R...]` | FP16 / BF16 / FP32 | 至少 1 维，且维度数不少于 `normalized_shape`。 |
| `normalized_shape` | `[R...]` | 必填 | Host 侧整数列表，指定末尾参与归一化的维度，不能为空。 |
| `weight` | `[R...]` | `None` | 可选缩放参数；dtype 与 `input` 相同或为 FP32，`weight` 与 `bias` 同时存在时两者 dtype 必须一致。 |
| `bias` | `[R...]` | `None` | 可选偏置参数，dtype 约束与 `weight` 相同。 |
| `eps` | — | float，`1e-5` | 防止除零的系数。 |
| `impl_mode` | — | int，0 | CANN 计算模式，原样传给 ACLNN。 |
| `impl` | — | `"auto"` | 仅关键字参数：`auto`、`cann_ops_transformer`、`torch_native`。 |

| 返回值 | 形状 | 数据类型 |
| --- | --- | --- |
| `out` | 与 `input` 相同 | 与 `input` 相同 |
| `mean` | `[A..., 1, ...]` | 与 `input` 相同 |
| `rstd` | `[A..., 1, ...]` | 与 `input` 相同 |

`mean` / `rstd` 在归一化维度上补 1，与输入维数相同；当不存在前置维度时形状为全 1。

## cann_ops_transformer 后端

- 调用 CANN toolkit 自带的 `aclnnLayerNormWithImplMode`，符号解析顺序与其他算子一致：
  先查 wheel 内置的 `libcust_opapi.so`，未找到时回退到 CANN toolkit 的 `libopapi.so`。
  不需要额外 vendor 包，也不需要设置 `ASCEND_CUSTOM_OPP_PATH`。
- `out`、`mean`、`rstd` 在 Host 侧先用 input 的 options 和 NPU format 分配，与旧绑定一致。
- `normalized_shape` 和 `eps` 是 Host 侧参数，不能传 NPU tensor。
- 本层不额外增加校验、类型转换或连续性拷贝；输入必须已经满足算子契约。
- 仅支持 arch22（A2/A3）。

## torch_native 后端

- 纯 PyTorch 参考实现：在 FP32 中计算，`out`、`mean`、`rstd` 再转回 input dtype。
- 按旧绑定的分配规则将 `mean` / `rstd` reshape 为 `[A..., 1, ...]`。
- `impl_mode` 只影响 CANN 的计算模式，参考实现忽略该参数。
- 与设备实现同样只在 arch22 注册。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import layer_norm

values = torch.randn(512, 4096, device="npu", dtype=torch.float16)
weight = torch.ones(4096, device="npu", dtype=torch.float32)
bias = torch.zeros(4096, device="npu", dtype=torch.float32)

out, mean, rstd = layer_norm(values, [4096], weight, bias, eps=1e-5)
print(out.shape)   # [512, 4096]
print(mean.shape)  # [512, 1]
```
