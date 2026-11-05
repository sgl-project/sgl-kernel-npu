# 打包 KV 缓存写入

## 硬件支持

| 实现 | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer_cann920` | 不支持 | 支持 |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 不支持 | 支持 |

## 接口

```python
from sgl_kernel_npu.operators.kvcacheio import kv_compress_epilog

def kv_compress_epilog(cache, x, slot_mapping, *, quant_group_size=64, quant_mode=None, round_scale=True, x_scale=1.0, impl='auto'):
    ...
```

`auto` 根据初始化的 CANN SDK 选择生产路径：9.1 使用旧绑定，9.2 使用正式绑定；不在执行失败后重试其他后端。

## 说明

仅 A5；64 元素量化组，输入宽度 448+64，先通过 get_npu_kv_quant_path() 确定分配格式。

分组量化 NoPE 并散写打包行，RoPE 尾部保持未量化；-1 槽跳过，0 槽有效。比较有效语义字节，不比较对齐填充。

## 参数

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| cache | Tensor | 必填 | Mutable paged cache; packed KV uint8/FP8 bytes, indexer FP8 payload |
| x | Tensor | 必填 | Input activation; shape described below; production device NPU |
| slot_mapping | Tensor | 必填 | int32 [T] destination slot; -1 skips a write |
| quant_group_size | int | 64 | 64 for migrated packed KV |
| quant_mode | int / str | None | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
| round_scale | bool | True | Scale rounding; CANN 9.1 normal FP8 ignores it |
| x_scale | float | 1.0 | Additional vendor quantization scale; migrated packed KV requires 1.0 |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

## 返回值

| 名称 | 类型 | 约束 |
| --- | --- | --- |
| cache | Tensor | The same mutated packed input cache. |

## cann_ops_transformer_cann920

使用 CANN 9.2 绑定，并拒绝不匹配的 SDK。auto 按缓存的 SDK 版本选择实现；A5 KV 格式也由同一版本确定。每个算子仅查找自身绑定。

## ascendc

Source: [sgl-kernel-npu device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp). The C++ wrapper directly launches the checked-in Ascend C kernel; this is not inferred from the Torch namespace. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

独立 eager PyTorch 参考；仅用于正确性检查，不自动替代生产实现，也不保证图捕获。元数据描述对象不等价于原生 tiling 字节。上述参考支持域外的模式会明确拒绝。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.kvcacheio import kv_compress_epilog

from sgl_kernel_npu.operators.kvcacheio import get_npu_kv_quant_path
path = get_npu_kv_quant_path()  # Resolve before allocation; A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
x = torch.randn(1, 512, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
kv_compress_epilog(cache, x, slots, impl="auto")
```

先构建安装当前 wheel 并初始化 SDK。上例只演示接口，不能替代 NPU 精度、性能或端到端验证。

```bash
python3 -m pytest tests/python/operators/kvcacheio/test_kv_compress_epilog.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[上游算子仓库](https://github.com/Ascend/ops-transformer)
