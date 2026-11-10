# mHC 后置混合

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 支持 |
| `cann_recipes_infer_cann910` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.hyper_conn import mhc_post_dpskv4

def mhc_post_dpskv4(x, residual, post, comb, *, impl='auto'):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

将层输出与 n 个残差流混合，公共参数顺序跨 SDK 保持一致。

`out[...,i,d] = post[...,i] × x[...,d] + Σ_j comb[...,j,i] × residual[...,j,d]`；参考以 FP32 累加后转为 x.dtype。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| residual | Tensor | 必填 | [T,n,C] residual streams |
| post | Tensor | 必填 | [T,n] post gates |
| comb | Tensor | 必填 | [T,n,n] residual matrix |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| output | Tensor | [T,n,C], x.dtype. |

## cann_ops_transformers_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnHcPost](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_hc_post.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import mhc_post_dpskv4

x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
residual = torch.randn(1, 4, 128, device="npu", dtype=x.dtype)
post = torch.ones(1, 4, device="npu", dtype=torch.float32)
comb = torch.eye(4, device="npu", dtype=torch.float32).unsqueeze(0)
y = mhc_post_dpskv4(x, residual, post, comb, impl="torch_native")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/hyper_conn/test_mhc_post_dpskv4.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
