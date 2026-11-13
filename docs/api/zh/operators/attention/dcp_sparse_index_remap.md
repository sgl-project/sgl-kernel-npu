# dcp_sparse_index_remap

## 硬件支持

| 实现 | arch22（A2） | arch22（A3） | arch35（A5） |
| --- | --- | --- | --- |
| `ascendc` | 未验证 | 未验证 | 不支持 |
| `torch_native` | 未验证 | 未验证 | 未验证 |

## 接口签名

```python
from sgl_kernel_npu.operators.attention import dcp_sparse_index_remap

dcp_sparse_index_remap(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
    *,
    impl: str = "auto",
) -> torch.Tensor
```

`impl` 默认为 `"auto"`，在 arch22 上选择 `"ascendc"`。显式指定 `"torch_native"` 可使用 PyTorch 实现。

## 功能描述

将解码上下文并行（DCP）的全局稀疏 token 索引压缩为当前 rank 的本地 KV 位置。设输入索引为 $i \geq 0$，页大小为 $P$，DCP 组大小为 $D$，则全局页号为 $p = \lfloor i/P \rfloor$。当 $p \bmod D = r$ 时，rank $r$ 保留该索引，并写入 $\lfloor p/D \rfloor P + i \bmod P$。

每行保留的索引按输入顺序排列，重复索引也会保留。负数输入会被忽略，剩余输出列填充 `-1`。

## 参数与返回值

| 参数 | 类型 | 含义与约束 |
| --- | --- | --- |
| `indices` | `torch.Tensor` | 连续的 int32 张量 `[rows, 2048]`。`ascendc` 要求张量位于 NPU 上。 |
| `dcp_size` | `int` | DCP 组大小，取 2 或 4。 |
| `dcp_rank` | `int` | 从 0 开始的 rank 编号，范围为 `[0, dcp_size)`。 |
| `page_size` | `int` | 每页 token 数，须为 `[1, 65536]` 内的 2 的幂。 |
| `impl` | `str` | 后端选择，默认为 `"auto"`。 |

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| output | `torch.Tensor` | 连续的 int32 张量，形状和设备与 `indices` 相同。 |

## ascendc 后端

Ascend C 向量算子在一次设备侧执行中完成页归属过滤、索引压缩和尾部填充。

## torch_native 后端

用 PyTorch 张量操作实现相同语义，也接受 CPU 张量。

## 使用示例

```python
import torch

from sgl_kernel_npu.operators.attention import dcp_sparse_index_remap

indices = torch.full((1, 2048), -1, dtype=torch.int32, device="npu")
indices[0, :3] = torch.tensor([128, 0, 130], dtype=torch.int32, device="npu")
local_indices = dcp_sparse_index_remap(indices, 4, 1, 128, impl="ascendc")
assert local_indices[0, :3].tolist() == [0, 2, -1]
```
