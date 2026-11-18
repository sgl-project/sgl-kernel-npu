# slot_map_lookup

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 不支持 |

## 函数签名

`auto` 选择 `ascendc`。

```python
from sgl_kernel_npu.operators.mem_cache import slot_map_lookup


def slot_map_lookup(slot_map, req_indices, topk_indices, block_dim=0, *, impl="auto"): ...
```

## 功能描述

查找每个请求的 top-k token 在设备缓存中的槽位。对有效的请求和 token 索引，读取 `slot_map[req_indices[b], topk_indices[b, k]]`，返回槽位和命中标记。槽位 `-1` 表示未命中。请求索引或 token 索引为负或越界时，返回标记 0 和位置 `-1`。

## 参数与返回值

输入张量必须连续且位于同一 NPU。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `slot_map` | Tensor | 必填 | INT32 `[slots, context]` 槽位映射，元素为非负槽号或 `-1`。`context` 为 8 的正整数倍。 |
| `req_indices` | Tensor | 必填 | INT32 `[batch]` 请求行索引。 |
| `topk_indices` | Tensor | 必填 | INT32 `[batch, 2048]` token 索引。 |
| `block_dim` | int | `0` | AscendC 使用的向量核数。0 表示使用设备全部可用向量核。 |
| `impl` | str | `"auto"` | 使用的实现。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `token_on_device` | Tensor | INT32 `[batch, 2048]` 命中标记，命中为 1，未命中为 0。 |
| `device_token_pos` | Tensor | INT32 `[batch, 2048]` 设备槽号，未命中为 `-1`。 |

## ascendc 后端

- 分块查找请求和 token 索引，并在当前 NPU 上写出两个结果。
- 正数 `block_dim` 受设备可用向量核数限制。

## torch_native 后端

- 用带掩码的 PyTorch 索引查找有效槽位，并生成整数命中标记。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import slot_map_lookup

slot_map = torch.full((2, 64), -1, dtype=torch.int32, device="npu")
slot_map[0, 7] = 12
req_indices = torch.tensor([0], dtype=torch.int32, device="npu")
topk_indices = torch.full((1, 2048), 7, dtype=torch.int32, device="npu")
hits, positions = slot_map_lookup(slot_map, req_indices, topk_indices, impl="ascendc")
```
