# move_intermediate_cache_kda

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mamba import move_intermediate_cache_kda


def move_intermediate_cache_kda(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
    *,
    impl="auto"
): ...
```

## 功能描述

跨层提交接受的 KDA 快照到持久状态，保留目标 head、V、K 的步幅。跳过负 last-step 项；空选择直接返回原目标。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ssm_states` | Tensor | `Required` | 可写浮点持久状态 [layers,pool,H,V,K]。 |
| `intermediate_state_cache` | Tensor | `Required` | 浮点 [layers,pool,steps,H,V,K]，H/V/K 内部连续。 |
| `dst_indices_tensor` | Tensor | `Required` | 整数目标槽 [requests]，并发请求目标应互不重复。 |
| `src_indices_tensor` | Tensor | `Required` | 整数源槽 [requests]。 |
| `last_steps_tensor` | Tensor | `Required` | 整数快照步数 [requests]，负值跳过请求。 |
| `h_block_size` | int | `1` | 头分块大小，不改变计算结果。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 原 ssm_states 张量，已原地更新。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import move_intermediate_cache_kda

state = torch.zeros(2, 4, 2, 128, 128, device="npu", dtype=torch.bfloat16)
snapshots = torch.randn(2, 4, 3, 2, 128, 128, device="npu", dtype=state.dtype)
slot = torch.tensor([1], device="npu", dtype=torch.int32)
step = torch.tensor([2], device="npu", dtype=torch.int32)
move_intermediate_cache_kda(state, snapshots, slot, slot, step, impl="triton")
```
