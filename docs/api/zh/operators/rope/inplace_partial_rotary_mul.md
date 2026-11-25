# 原地部分 RoPE

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 支持 |
| `cann_recipes_infer_cann910` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

def inplace_partial_rotary_mul(x, cos, sin, *, rotary_mode='interleave', partial_slice=None, impl='auto'):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

仅原地修改选定特征切片；比较实现前须克隆 x。

旋转对 `(a,b)` 变为 `(a cos − b sin, b cos + a sin)`；rotary_mode 控制交错或半区配对，其余特征保持不变。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| cos | Tensor | 必填 | Broadcast-compatible cosine factors for selected rotary features |
| sin | Tensor | 必填 | Broadcast-compatible sine factors, matching cos |
| rotary_mode | str / int | 'interleave' | RoPE public API: interleave or half; compressor uses native integer mode |
| partial_slice | sequence[int] / None | None | [begin,end), None selects the full last dimension; even rotary width |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| x | Tensor | The same mutated input tensor. |

## cann_ops_transformers_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnInplacePartialRotaryMul](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/inplace_partial_rotary_mul.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

x = torch.randn(1, 1, 128, device="npu", dtype=torch.bfloat16)
cos = torch.ones(1, 1, 64, device="npu", dtype=x.dtype)
sin = torch.zeros_like(cos)
y = inplace_partial_rotary_mul(x, cos, sin, partial_slice=[64, 128], impl="torch_native")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/rope/test_inplace_partial_rotary_mul.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
