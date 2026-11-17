# mHC 前置混合

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 支持 |
| `cann_recipes_infer_cann910` | 支持 | 支持 |
| `cann_recipes_infer_cann920` | 不支持 | 支持（DSV4.1） |
| `torch_native` | 支持 | 支持 |

`cann_recipes_infer_cann920` 还要求已初始化的 CANN SDK 为 9.2.x；该实现只在 arch35 注册，arch22 上显式选择会抛出 `UnsupportedGenerationError`。

## 接口

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

def mhc_pre_dpskv4(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps,
                   *, pre_mix=None, return_pre=False, impl='auto'):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。DSV4.1 的融合 kernel 需显式选择 `cann_recipes_infer_cann920`。

## 说明

输入 [T,n,C]，不融合模型 input layernorm。

对展平流进行 RMS 归一化投影，生成 sigmoid pre/post gate 和 Sinkhorn 归一化残差矩阵；`y = Σ_i pre_i × x_i`。y 保留输入 dtype，参考 post/comb 为 FP32。

`pre_mix` 传入时直接作为 pre 权重参与 y 累加，不再由投影结果计算；`return_pre=True` 时把 pre 作为第四个返回值追加在 comb 之后。这两个参数仅 `cann_recipes_infer_cann920` 与 `torch_native` 支持。

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
| pre_mix | Tensor | None | 仅限关键字；[T,n] FP32 pre 权重，要求 `cann_recipes_infer_cann920` 或 `torch_native` |
| return_pre | bool | False | 仅限关键字；为 True 时多返回一个 pre 输出 |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

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

DSV4.1 融合路径，仅 CANN 9.2.x + A5；`npu_hc_pre_v2` 只随 arch35 构建进入 wheel。调用 wheel 内捆绑的 `torch.ops.sgl_cann_recipes_infer.npu_hc_pre_v2`，支持 `pre_mix` 输入与 pre 输出；绑定缺失时明确报错，不回退其他后端。

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_pre_dpskv4

x = torch.randn(1, 4, 128, device="npu", dtype=torch.bfloat16)
fn = torch.randn(24, 512, device="npu", dtype=torch.bfloat16)
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)
y, post, comb = mhc_pre_dpskv4(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl="torch_native")

# DSV4.1 融合 kernel：需要 CANN 9.2.0 wheel（含 recipes vendor）。
y, post, comb, pre = mhc_pre_dpskv4(
    x, fn, scale, base, 4, 20, 1e-6, 1e-6,
    return_pre=True, impl="cann_recipes_infer_cann920",
)
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/hyper_conn/test_mhc_pre_dpskv4.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

`cann_recipes_infer_cann920` 的 DSV4.1 用例只在 CANN 9.2.x + arch35 上执行，其余环境自动跳过；基线用例走 `auto`，由 SDK 版本决定后端。

[上游算子仓库](https://github.com/Ascend/ops-transformer)
