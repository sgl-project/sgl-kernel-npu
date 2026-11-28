# copy_tracked_conv_states

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mamba import copy_tracked_conv_states


def copy_tracked_conv_states(
    conv_states, src_slots, dst_slots, track_steps, *, impl="auto"
): ...
```

## 功能描述

在投机解码的 Radix Cache 检查点追踪阶段，单次启动完成跨层条件拷贝：仅对满足 `track_steps[b] >= 0`、`src_slots[b] >= 0`、`dst_slots[b] >= 0` 且 `src_slots[b] != dst_slots[b]` 的请求执行 `conv_states[:, dst_slots[b]] = conv_states[:, src_slots[b]]`。其余未追踪行直接在 kernel 内跳过，避免全 batch 跨层自拷贝。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `conv_states` | `torch.Tensor` | `[layers, pool, ...]`（>=2D） | 任意数值类型 | 必填 | 跨层卷积状态池，原地更新。 |
| `src_slots` | `torch.Tensor` | `[B]` | 整型 | 必填 | 源槽位索引（`< 0` 跳过）。 |
| `dst_slots` | `torch.Tensor` | `[B]` | 整型 | 必填 | 目标槽位索引（`< 0` 或等于源槽位时跳过）。 |
| `track_steps` | `torch.Tensor` | `[B]` | 整型 | 必填 | 追踪步索引（`< 0` 表示该请求本步无需保存检查点）。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `result` | `torch.Tensor` | 原地更新后的 `conv_states` 别名。 |

## triton 后端

- 对连续张量采用 56-AIV 持久化 grid（`BLOCK_E=4096`、`tl.int64` 偏移），单次启动覆盖所有层与请求。
- 若 `conv_states` 非连续，自动仅对有效行执行索引拷贝以保持原始步幅和语义。

## torch_native 后端

- 独立的逐请求 PyTorch 切片拷贝参考实现，拷贝前先克隆源切片避免重叠覆盖。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import copy_tracked_conv_states

conv_states = torch.randn(48, 72, 10, 10240, device="npu", dtype=torch.bfloat16)
src_slots = torch.tensor([0, 1, 2, 3], device="npu", dtype=torch.int64)
dst_slots = torch.tensor([10, 11, 12, 13], device="npu", dtype=torch.int64)
track_steps = torch.tensor([2, -1, 0, -1], device="npu", dtype=torch.int64)
copy_tracked_conv_states(conv_states, src_slots, dst_slots, track_steps, impl="auto")
```
