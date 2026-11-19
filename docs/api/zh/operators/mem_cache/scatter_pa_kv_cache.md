# scatter_pa_kv_cache

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.mem_cache import scatter_pa_kv_cache


def scatter_pa_kv_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    *,
    compress_lens=None,
    compress_seq_offsets=None,
    seq_lens=None,
    cache_mode="PA_NZ",
    impl="auto"
): ...
```

## 功能说明

通过 slot_mapping 将 K/V 行写入传入的分页缓存。未压缩 ND 缓存的每个槽位接收对应的 K/V 行。SDK 还支持压缩缓存模式。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `key`, `value` | Tensor | [T, Hkv, D] for Norm, FP16，用于 example | 必传 | NPU 上待写入的缓存行。 |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, D] for Norm, 输入 dtype | 必传 | 由调用方分配的分页目标缓存，原地更新。 |
| `slot_mapping` | Tensor | [T], INT32，用于 example | 必传 | 各 token 行对应的物理槽位。 |
| `compress_lens`, `compress_seq_offsets`, `seq_lens` | Tensor or None | SDK metadata tensors | `None` | 压缩缓存放置的可选元数据。 |
| `cache_mode` | str | - | `'PA_NZ'` | 缓存布局模式。示例的未压缩 ND 缓存使用 Norm，厂商默认格式为 PA_NZ。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `result` | None | - | 结果写入传入的缓存。 |

## torch_npu 后端

调用 `torch_npu.npu_scatter_pa_kv_cache`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_scatter_pa_kv_cache.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import scatter_pa_kv_cache

key = torch.randn((3, 2, 128), dtype=torch.float16, device="npu")
value = torch.randn_like(key)
key_cache = torch.zeros((2, 16, 2, 128), dtype=key.dtype, device=key.device)
value_cache = torch.zeros_like(key_cache)
slots = torch.tensor([31, 15, 16], dtype=torch.int32, device=key.device)
scatter_pa_kv_cache(
    key, value, key_cache, value_cache, slots, cache_mode="Norm", impl="torch_npu"
)
```
