# mHC 前置混合

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 支持 |
| `cann_ops_transformers_cann920_v2` | 支持 | 支持 |
| `cann_recipes_infer_cann910` | 支持 | 支持 |
| `cann_recipes_infer_cann920`（V2 兼容名称） | | 支持 |
| `torch_native` | 支持 | 支持 |

DeepSeek V4.1 路径需要安装包含目标架构融合后端的 CANN 9.2 wheel。

## 接口

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

def mhc_pre_dpskv4(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps,
                   *, pre_mix=None, return_pre=False, impl='auto'):
    ...
```

传入 `pre_mix` 或设置 `return_pre=True` 时，`auto` 在 A3 和 A5 上都选择 CANN MhcPreSinkhornV2。两项均未指定时，按初始化的 SDK 选择 DeepSeek V4 路径。

### V2 直接入口

`mhc_pre_sinkhorn_v2(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps, *, pre_mix, impl='auto')` 在 A3 和 A5 上均调用 wheel 内的 V2 算子并返回 `(y, post, comb, pre)`。`pre_mix` 是上一子层的权重，返回的 `pre` 由当前残差计算。

## 说明

输入 [T,n,C]，不融合模型 input layernorm。

对展平流进行 RMS 归一化投影，生成 sigmoid pre/post gate 和 Sinkhorn 归一化残差矩阵；`y = Σ_i pre_i × x_i`。y 保留输入 dtype，参考 post/comb 为 FP32。

`pre_mix` 传入时直接作为 y 的加权系数。`return_pre=True` 时追加根据当前残差计算的 pre，包括传入 `pre_mix` 的情况。A3 和 A5 的融合后端及 `torch_native` 均支持这两个参数。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| fn | Tensor | 必填 | [n*n+2*n,n*C] projection matrix |
| scale | Tensor | 必填 | [3] gate/matrix scales |
| base | Tensor | 必填 | [n*n+2*n] projection bias |
| hc_mult | int | 必填 | Number n of residual streams |
| sinkhorn_iters | int | 必填 | Positive Sinkhorn iteration count |
| hc_eps | float | 必填 | Gate and Sinkhorn stability epsilon |
| rms_norm_eps | float | 必填 | Internal projection RMS epsilon, not the model layernorm |
| pre_mix | Tensor | None | 仅限关键字，参与 y 累加的 [T,n] FP32 权重 |
| return_pre | bool | False | 仅限关键字；为 True 时多返回一个 pre 输出 |
| impl | str | 'auto' | 已注册的实现，或用 `auto` 自动选择生产路径 |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| y | Tensor | [T,C], x.dtype. |
| post | Tensor | [T,n], reference FP32. |
| comb | Tensor | [T,n,n], reference FP32. |
| pre | Tensor | [T,n] FP32；仅 `return_pre=True` 时返回。 |

## cann_ops_transformers_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。不支持 pre 输出。

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnHcPre](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_pre.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## cann_recipes_infer_cann920

这是旧版 A5 `mhc_pre_dpskv4` 调用方的兼容名称，调用 wheel 内相同的 V2 算子。

## cann_ops_transformers_cann920_v2

A3 和 A5 使用 wheel 内的 MhcPreSinkhornV2 和 HF32。打包时将其注册为 `aclnnSglMhcPreSinkhornV2`，使 DeepSeek V4 和 V4.1 的算子可以同时加载。此后端使用 4 条残差流和 20 次 Sinkhorn 迭代，隐藏维度需按 128 对齐。[CANN 算子约束](https://gitcode.com/cann/ops-transformer/blob/53f5c56996be158b52579d9cdd484b54a38dfb64/mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhornV2.md)

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4, mhc_pre_sinkhorn_v2

x = torch.randn(1, 4, 5120, device="npu", dtype=torch.bfloat16)
fn = torch.randn(24, 4 * 5120, device="npu", dtype=torch.float32)
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_dpskv4(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl="torch_native")

# DeepSeek V4.1 使用上一子层的 pre 权重。
pre_mix = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=x.device)
y, post, comb, pre = mhc_pre_sinkhorn_v2(
    x, fn, scale, base, 4, 20, 1e-6, 1e-6,
    pre_mix=pre_mix,
)
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/hyper_conn/test_mhc_pre_dpskv4.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

安装的 wheel 包含对应架构的融合后端时，执行 DeepSeek V4.1 用例。

[上游算子仓库](https://github.com/Ascend/ops-transformer)
