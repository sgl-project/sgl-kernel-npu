# hc_post

上游文档：[recipes npu_hc_post](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/docs/custom-npu_hc_post.md)。

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `cann_recipes_infer_cann910` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认 `auto` 在 CANN 9.1.0 下选择 `cann_recipes_infer_cann910`，其他 CANN 版本会报错。

```python
from sgl_kernel_npu.operators.hyper_conn import hc_post

def hc_post(x, residual, post, comb, *, impl="auto"):
    ...
```

## 功能说明

混合各路残差，并加上当前层输出的加权结果。对于输出残差流 `j`、隐藏维度通道 `d`：

$$
Y_{\ldots,j,d}
= P_{\ldots,j}X_{\ldots,d}
+ \sum_{i=0}^{3} C_{\ldots,i,j}R_{\ldots,i,d}
$$

`X`、`R`、`P`、`C` 分别对应 `x`、`residual`、`post`、`comb`。算子沿 `comb` 的第一个残差流维度求和，混合矩阵直接传入，无需转置。输入不会被修改。

## 参数与返回值

所有输入必须连续，并位于同一张 NPU 上。`T` 为 token 数，也可以保留 `[B,S]` 两个批次、序列维度。各维度必须为正数，残差流数固定为 4，`D` 只能为 4096 或 7168。

| 参数 | 类型 / shape / dtype | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `x` | Tensor `[T,D]` 或 `[B,S,D]`；float16、bfloat16 或 float32 | 必填 | 当前层输出。 |
| `residual` | Tensor `[T,4,D]` 或 `[B,S,4,D]`；dtype 与 `x` 相同 | 必填 | 各路残差；前置维度和 `D` 必须与 `x` 一致。 |
| `post` | Tensor `[T,4]` 或 `[B,S,4]`；float16、bfloat16 或 float32 | 必填 | 每路输出中 `x` 的权重；前置维度必须与 `x` 一致。 |
| `comb` | Tensor `[T,4,4]` 或 `[B,S,4,4]`；dtype 与 `post` 相同 | 必填 | 残差混合权重，最后两个维度依次表示输入流和输出流；前置维度必须与 `x` 一致。 |
| `impl` | `str`，仅关键字传入 | `"auto"` | `auto`、`cann_recipes_infer_cann910` 或 `torch_native`。 |

| 返回值 | 类型 / shape / dtype | 含义 |
| --- | --- | --- |
| `output` | Tensor，shape、dtype 和 device 与 `residual` 相同 | 新分配的混合残差。 |

## cann_recipes_infer_cann910 后端

封装 recipes 的 `npu_hc_post` 接口。使用 CANN 9.1.0，以及构建时包含 `cann_recipes_infer` profile 的 sgl-kernel-npu wheel。所需内核和预编译 Torch 绑定随 wheel 提供，无需额外安装 recipes。

## torch_native 后端

使用 PyTorch 运算，以 float32 计算中间结果，最后转换为 `x.dtype`，shape 和 dtype 约束保持一致。在其他 CANN 版本上需要显式选择；`auto` 不会自动回退。浮点累加结果可能与 recipes 实现存在差异。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.hyper_conn import hc_post

x = torch.randn(1, 4096, device="npu", dtype=torch.bfloat16)
residual = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
post = torch.randn(1, 4, device="npu", dtype=torch.float32)
comb = torch.randn(1, 4, 4, device="npu", dtype=torch.float32)

output = hc_post(x, residual, post, comb)  # CANN 9.1.0，wheel 已包含 recipes
output_native = hc_post(x, residual, post, comb, impl="torch_native")
print(output.shape)  # torch.Size([1, 4, 4096])
```
