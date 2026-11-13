# dcp_verify_page_tables

## 硬件支持

| 实现 | arch22（A2） | arch22（A3） | arch35（A5） |
| --- | --- | --- | --- |
| `ascendc` | 未验证 | 未验证 | 不支持 |
| `torch_native` | 未验证 | 未验证 | 未验证 |

## 接口签名

```python
from sgl_kernel_npu.operators.attention import dcp_verify_page_tables

dcp_verify_page_tables(
    slots: torch.Tensor,
    requests: torch.Tensor,
    ordinary: torch.Tensor,
    local: torch.Tensor,
    active_cols: int,
    query_len: int,
    page_size: int,
    dcp_size: int,
    *,
    impl: str = "auto",
) -> None
```

`impl` 默认为 `"auto"`，在 arch22 上选择 `"ascendc"`。显式指定 `"torch_native"` 可使用 PyTorch 实现。

## 功能描述

生成 Target Verify 所需的普通页表与 rank 本地页表。设批次第 $i$ 行的请求池行号为 $s_i$，页大小为 $P$，DCP 组大小为 $D$，每个请求的 query 数为 $Q$。对每个有效普通页列 $j$，写入 $\mathrm{ordinary}[i,j] = \lfloor \mathrm{slots}[s_i,jP]/P \rfloor$。本地页表第 $k$ 列取自普通页表第 $kD$ 列，对每个 $0 \leq q < Q$ 写入 $\mathrm{local}[iQ+q,k] = \lfloor \mathrm{ordinary}[i,kD]/D \rfloor$。

算子原地覆盖两个缓冲区。未使用的列和非法请求编号对应的所有输出行均置零，即使本次调用的有效范围缩小。

## 参数与返回值

| 参数 | 类型 | 含义与约束 |
| --- | --- | --- |
| `slots` | `torch.Tensor` | 连续的 int32 张量 `[pool_rows, slot_cols]`，将请求内的 token 映射到物理 KV 槽位。 |
| `requests` | `torch.Tensor` | 连续的 int32 或 int64 张量 `[rows]`，表示请求池行号。超出 `[0, pool_rows)` 的行号对应全零输出。 |
| `ordinary` | `torch.Tensor` | 调用方预分配的连续 int32 张量 `[rows, ordinary_cols]`。 |
| `local` | `torch.Tensor` | 调用方预分配的连续 int32 张量 `[rows * query_len, local_cols]`。 |
| `active_cols` | `int` | 有效普通页数，范围为 `[0, min(ceil(slot_cols / page_size), ordinary_cols, local_cols * dcp_size)]`。 |
| `query_len` | `int` | 每个请求的 query 数，须大于 0。 |
| `page_size` | `int` | 每页 token 数，须为 `[1, 65536]` 内的 2 的幂。 |
| `dcp_size` | `int` | DCP 组大小，取 2 或 4。 |
| `impl` | `str` | 后端选择，默认为 `"auto"`。 |

| 返回值 | 类型 | 含义 |
| --- | --- | --- |
| — | `None` | 原地写入 `ordinary` 和 `local`。 |

所有张量须位于同一设备。`ascendc` 要求张量位于 NPU 上。两个输出缓冲区各自均不可与其他实参重叠。

## ascendc 后端

Ascend C 向量算子在一次设备侧执行中读取槽位映射，并将两个页表写入调用方提供的缓冲区。

## torch_native 后端

用 PyTorch 张量操作实现相同语义，也接受 CPU 张量。

## 使用示例

```python
import torch

from sgl_kernel_npu.operators.attention import dcp_verify_page_tables

slots = torch.tensor([[8, 9, 16, 17, 24, 25]], dtype=torch.int32, device="npu")
requests = torch.tensor([0], dtype=torch.int64, device="npu")
ordinary = torch.empty((1, 4), dtype=torch.int32, device="npu")
local = torch.empty((2, 2), dtype=torch.int32, device="npu")
dcp_verify_page_tables(slots, requests, ordinary, local, 3, 2, 2, 2, impl="ascendc")
assert ordinary.tolist() == [[4, 8, 12, 0]]
assert local.tolist() == [[2, 6], [2, 6]]
```
