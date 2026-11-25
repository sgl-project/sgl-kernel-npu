# mhc_pre_sinkhorn

## 芯片支持

| 实现 | arch22（A3） | arch35（950DT） |
| --- | --- | --- |
| cann_ops_transformer | 已验证 | 已验证 |
| cann_ops_transformer_v2 | wheel 内置 vendor | — |
| torch_native | 已在 NPU 验证 | 已在 NPU 验证 |

CANN 9.1.0 在 A3 和 Ascend950 上支持 hidden_size=128、256 等小尺寸；CANN 9.2.0 在 Ascend950 上仅支持 hidden_size=4096 或 7168。GLM-5.3-Flash 的 hidden_size=4096 可用于两个版本。

## 函数原型

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_sinkhorn

def mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult=4, num_iters=20, hc_eps=1e-6, norm_eps=1e-6, *, return_pre=False, pre_mix=None, impl="auto"):
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
| `return_pre` | bool | 为 True 时追加 FP32 hpre [T,N]（sigmoid + hc_eps） | False |
| `pre_mix` | Tensor 或 None | 同设备 FP32 [T,N]，用于计算 hin，本层系数仍由 x 预测。支持 `cann_ops_transformer_v2` 和 `torch_native`。 | None |
| impl | str | auto / cann_ops_transformer / cann_ops_transformer_v2 / torch_native | auto |

适配层不额外限制输入类型、步长、设备、形状或标量值，支持范围与错误检查交由所选后端决定。示例采用 GLM 的 N=4、FP16/BF16 激活与 FP32 参数。空输入/IDLE 仍由 SGLang 调用方按 glmx PR #21 的约定短路处理，适配层不新增该行为。

| 返回 | 类型 | 形状与类型 |
| --- | --- | --- |
| output | tuple[Tensor,Tensor,Tensor] | (hin [T,D] in vendor dtype for cann_ops_transformer, hpost [T,N] FP32, hres [T,N*N] FP32) |

默认返回三项。设置 `return_pre=True` 时追加 hpre，供 V4.1 下一子层加权混合。未提供 `pre_mix` 时，CANN 后端通过可选的反向辅助输出来取得 hpre。

## cann_ops_transformer

首次执行时加载 wheel 内置绑定。含 V2 vendor 的 wheel 使用该 vendor 处理省略 `pre_mix` 的调用，返回值兼容旧接口。其余 wheel 继续通过 BSND 绑定执行，将 [T,N,D] 扩为 [T,1,N,D]，并在返回时去掉序列维。

## cann_ops_transformer_v2

A3 的 `ops_transformer` 和 `ops_transformer_dsv4` 两个构建 profile 均打包同一个 mHC pre vendor。绑定使用 TND 布局并开启 HF32。传入 `pre_mix` 且设置 `return_pre=True` 时，返回本层 hpre，省去反向辅助张量分配。[固定版本 ACLNN 接口](https://gitcode.com/cann/ops-transformer/blob/53f5c56996be158b52579d9cdd484b54a38dfb64/mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2.md)

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

# A3 V4.1 使用上一子层的 FP32 权重。
previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="npu")
y, post, comb, pre = mhc_pre_sinkhorn(
    x, phi, alpha, bias, pre_mix=previous, return_pre=True,
    impl="cann_ops_transformer_v2",
)
```
