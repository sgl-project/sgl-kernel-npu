# reshape_and_cache

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.mem_cache import reshape_and_cache


def reshape_and_cache(
    key, value, key_cache, value_cache, slot_indices, *, impl="auto"
): ...
```

## 功能说明

将输入键值行写入物理分页槽位。设页大小为 L，槽位 s 指向页 s//L 中的第 s%L 行。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `key`, `value` | Tensor | [T, Hkv, D] 在示例中, FP16 | 必传 | 与目标缓存同卡的 T 个缓存行。 |
| `key_cache`, `value_cache` | Tensor | [P, L, Hkv, D] 在 ND 布局下, 输入 dtype | 必传 | 由调用方分配的分页目标缓存，原地更新。 |
| `slot_indices` | Tensor | [T], INT32 在示例中 | 必传 | 各输入行对应的物理槽位。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `result` | None | - | 结果写入传入的缓存。 |

## torch_npu 后端

调用 `torch_npu._npu_reshape_and_cache`。形状、dtype 和布局组合以已安装的 SDK 为准。 要求 SDK 提供 ATB 私有 reshape-and-cache 入口。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/atb_ops.yaml)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import reshape_and_cache

key = torch.randn((3, 2, 128), dtype=torch.float16, device="npu")
value = torch.randn_like(key)
key_cache = torch.zeros((2, 16, 2, 128), dtype=key.dtype, device=key.device)
value_cache = torch.zeros_like(key_cache)
slots = torch.tensor([31, 15, 16], dtype=torch.int32, device=key.device)
reshape_and_cache(key, value, key_cache, value_cache, slots, impl="torch_npu")
```
