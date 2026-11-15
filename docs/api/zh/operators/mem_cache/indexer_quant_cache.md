# 索引器量化缓存写入

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | 支持 | 支持 |
| `cann_recipes_infer_cann910` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.mem_cache import indexer_quant_cache

def indexer_quant_cache(cache, scale_cache, x, slot_mapping, *, impl='auto', **kwargs):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

参考支持有限值、每行不超过 128 元素的普通 FP8；A3 可用性取决于所安装的 kernel 二进制。

`scale = max(abs(x))/448`，`payload = cast_fp8(clamp(x/scale))`。CANN 9.1 使用未取整 FP32 scale 和除法；9.2 支持 round_scale（向上取 2 的幂）并使用 FP32 倒数乘法。全零行 scale=0；x_scale 仅用于 HiFloat8。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| cache | Tensor | 必填 | Mutable paged cache; packed KV uint8/FP8 bytes, indexer FP8 payload |
| scale_cache | Tensor | 必填 | Mutable FP32 indexer block scales, one per cache row |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| slot_mapping | Tensor | 必填 | int32 [T] destination slot; -1 skips a write |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### 关键字参数

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| quant_mode / round_scale / x_scale | str / bool / float | Defaults fp8 / True / 1.0; reference supports normal FP8 only. |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| cache, scale_cache | Tensor tuple | The same mutated input buffers. |

## cann_ops_transformers_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。

## cann_recipes_infer_cann910

Source: [cann-recipes-infer; aclnnIndexerCompressEpilog](https://gitcode.com/cann/cann-recipes-infer/blob/35a476c2b1ea61990380a7115ea991023aedc7d0/ops/ascendc/torch_ops_extension/custom_ops/csrc/npu_indexer_compress_epilog.cpp). The recipes custom_ops bridge calls the indicated recipes ACLNN operator. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import indexer_quant_cache

# empty avoids 不支持 NPU zero_ on FP8; the selected row is overwritten.
cache = torch.empty(1, 128, 1, 128, device="npu", dtype=torch.float8_e4m3fn)
scales = torch.empty(1, 128, 1, 1, device="npu", dtype=torch.float32)
x = torch.randn(1, 128, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
indexer_quant_cache(cache, scales, x, slots, impl="auto")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/mem_cache/test_indexer_quant_cache.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
