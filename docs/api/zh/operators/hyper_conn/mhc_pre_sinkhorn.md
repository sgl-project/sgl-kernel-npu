# mhc_pre_sinkhorn

## 芯片支持

| 实现 | arch22（A3） | arch35（950DT） |
| --- | --- | --- |
| cann_ops_transformer | 已验证 | 已验证 |
| torch_native | 已在 NPU 验证 | 已在 NPU 验证 |

CANN 9.1.0 在 A3 和 Ascend950 上支持 hidden_size=128、256 等小尺寸；CANN 9.2.0 在 Ascend950 上仅支持 hidden_size=4096 或 7168。GLM-5.3-Flash 的 hidden_size=4096 可用于两个版本。

## 函数原型

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

def mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult=4, num_iters=20, hc_eps=1e-6, norm_eps=1e-6, *, impl="auto"):
    ...
```

`auto` 固定选择 `cann_ops_transformer`，不回退。

## 功能说明

在 FP32 中对展开的残差流做 RMS 归一化并用 phi 投影；pre/post 门值使用 sigmoid，post 固定乘 2。残差矩阵交替按行列做 Sinkhorn 归一化。未融合输出 RMSNorm。

设 u = flatten(x)，z = (u / sqrt(mean(u**2) + norm_eps)) @ phi.T。按 pre/post/res 拆分 z，分别乘 alpha[0:3] 并加对应 bias。

## 参数与返回值

| 参数 | 类型 | 约束 | 默认 |
| --- | --- | --- | --- |
| x | Tensor | [T,N,D], FP16/BF16 | 必填 |
| phi | Tensor | [(2+N)*N,N*D], FP32 | 必填 |
| alpha | Tensor | [3], FP32 | 必填 |
| bias | Tensor | [(2+N)*N], FP32 | 必填 |
| hc_mult | int | 残差流数量 N，支持范围由后端决定 | 4 |
| num_iters | int | Sinkhorn 迭代次数，支持范围由后端决定 | 20 |
| hc_eps | float | 透传后端 | 1e-6 |
| norm_eps | float | 透传后端 | 1e-6 |
| impl | str | auto / cann_ops_transformer / torch_native | auto |

适配层不额外限制输入类型、步长、设备、形状或标量值，支持范围与错误检查交由所选后端决定。示例采用 GLM 的 N=4、FP16/BF16 激活与 FP32 参数。空输入/IDLE 仍由 SGLang 调用方按 glmx PR #21 的约定短路处理，适配层不新增该行为。

| 返回 | 类型 | 形状与类型 |
| --- | --- | --- |
| output | tuple[Tensor,Tensor,Tensor] | (hin [T,D] in vendor dtype for cann_ops_transformer, hpost [T,N] FP32, hres [T,N*N] FP32) |

## cann_ops_transformer

依赖 CANN 9.1 的公开 Python API，首次执行时加载缓存；错误直接上抛，不回退旧算子。不设置额外架构白名单。

适配层将 [T,N,D] 扩为 [T,1,N,D] 的 BSND 布局，返回时去掉序列维。保留 vendor 输出类型，激活类型转换留给 SGLang 调用方，与 GitCode glmx PR #21 一致。

## torch_native

残差投影按行 softmax 后加 hc_eps，再除以列和加 hc_eps；随后交替除以行和加 hc_eps、列和加 hc_eps，共 num_iters-1 轮。hin = sum_i (sigmoid(pre_i)+hc_eps)*x_i。

FP32 中间计算，激活输出转为输入类型。独立精度基准，不作为自动回退。

CANN 使用 HF32 投影。在 Ascend910_9362、CANN 9.1 环境下，FP32 post/comb 的比较容差为 atol=5e-5、rtol=5e-5，基于九个随机种子、FP16/BF16 输入、两组 eps 和五组非空随机输入 shape 校准。FP16 激活使用 atol=1e-5、rtol=2e-3，BF16 使用 atol=1e-5、rtol=1e-2，避免共享绝对容差 0.05 掩盖小幅激活上的错误；相同阈值已在上述两台设备通过验证。

输入使用借用模型配置尺寸的构造数据，不是捕获激活或 checkpoint 权重；通过不能视为整网精度验收。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

x = torch.randn(1, 4, 4096, device="npu", dtype=torch.bfloat16)
phi = torch.randn(24, 16384, device="npu", dtype=torch.float32) / 128
alpha = torch.ones(3, device="npu", dtype=torch.float32)
bias = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_sinkhorn(x, phi, alpha, bias, impl="cann_ops_transformer")
```
