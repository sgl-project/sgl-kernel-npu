# mhc_post

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| cann | 待硬件验证 | 待硬件验证 |
| torch_native | 待硬件验证 | 待硬件验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_post

def mhc_post(x, h_res, h_out, h_post, *, impl="auto"):
    ...
```

`auto` 固定选择 `cann`，不回退。

## 功能说明

在 FP32 中计算 y[t,i,d] = h_post[t,i] * h_out[t,d] + sum_j h_res[t,i,j] * x[t,j,d]，最终转为 x.dtype。

## 参数与返回值

| 参数 | 类型 | 约束 | 默认 |
| --- | --- | --- | --- |
| x | Tensor | [T,N,D], FP16/BF16 | 必填 |
| h_res | Tensor | [T,N,N], FP32 | 必填 |
| h_out | Tensor | [T,D], 与 x 类型相同 | 必填 |
| h_post | Tensor | [T,N], FP32 | 必填 |
| impl | str | auto / cann / torch_native | auto |

适配层不额外限制输入类型、步长、设备、形状或标量值，支持范围与错误检查交由所选后端决定。示例采用 GLM 的 N=4、FP16/BF16 激活与 FP32 参数。T=0 不启动 CANN，返回空 Tensor（与 x 类型相同）。

| 返回 | 类型 | 形状与类型 |
| --- | --- | --- |
| output | Tensor | y [T,N,D]，cann 保留 vendor 输出类型 |

## cann

依赖 CANN 9.1 的公开 Python API，首次非空执行时加载缓存；错误直接上抛，不回退旧算子。不设置额外架构白名单，但不代表已经确认硬件支持或完成 NPU 验证。

参数顺序与 vendor API 一致：残差流、残差矩阵、层输出、post 门值。

## torch_native

使用独立的 torch.bmm 残差映射和 post 门值相加，不调用 vendor 算子。

FP32 中间计算，激活输出转为输入类型。独立精度基准，不作为自动回退；精度阈值与性能待 NPU 验收。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_post

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
h_res = torch.eye(4, device="npu", dtype=torch.float32).unsqueeze(0)
h_out = torch.zeros(1, 4096, device="npu", dtype=x.dtype)
h_post = torch.ones(1, 4, device="npu", dtype=torch.float32)
y = mhc_post(x, h_res, h_out, h_post, impl="cann")
```
