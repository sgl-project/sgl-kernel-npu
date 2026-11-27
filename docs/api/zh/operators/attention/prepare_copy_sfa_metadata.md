# prepare_copy_sfa_metadata

## 硬件支持

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| triton | 未验证 | 未验证 |
| torch_native | 未验证 | 未验证 |

## 接口

`auto` 选择 Triton。

```python
from sgl_kernel_npu.operators.attention import prepare_copy_sfa_metadata

def prepare_copy_sfa_metadata(
    *,
    source_ids: torch.Tensor,
    valid_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    request_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    capacity: int,
    query_width: int = 1,
    impl="auto",
) -> None:
    ...
```

## 说明

原地准备 Copy-SFA 长度和 HBM 页表，并将无效请求的源索引设为 -1。

## 参数及返回值

| 参数 | 形状、类型和含义 |
| --- | --- |
| `source_ids` | [B * query_width, K] int32，源索引，无效行被修改 |
| `valid_rows` | [B] bool，有效输入标记 |
| `sparse_rows` | [B] bool，稀疏输入标记 |
| `request_slots` | [B] int32，请求槽位 |
| `seq_lens` | [B] int32，活动长度 |
| `actual_seq_lengths_kv` | [B] int32，输出长度 |
| `hbm_block_table` | [B, H] int32，输出 HBM 页表 |
| `capacity` | int，缓存容量 |
| `query_width` | int，每请求查询数，默认 1 |
| `impl` | str，默认 auto，可选 triton、torch_native |

返回 `None`；输出张量原地修改。除注明外张量应连续并位于同一设备。

源索引有效行保持原值，无效行写 -1。有效 HBM 页表第 j 列为请求槽位乘 H 加 j，无效行为 j。有效稀疏行长度为 capacity + query_width - 1，非稀疏行为 seq_lens，无效行为零。B、Q、K、H 必须为正，K 必须为 2 的幂以满足 Triton arange，且 K >= H。张量须连续。

## triton backend

使用 Triton 内核。

## torch_native backend

使用 PyTorch 操作作为参考实现。

## 示例

```python
import torch
from sgl_kernel_npu.operators.attention import prepare_copy_sfa_metadata

prepare_copy_sfa_metadata(
    source_ids=torch.zeros((2, 16), dtype=torch.int32, device="npu"),
    valid_rows=torch.tensor([True, False], device="npu"),
    sparse_rows=torch.tensor([True, False], device="npu"),
    request_slots=torch.tensor([1, 2], dtype=torch.int32, device="npu"),
    seq_lens=torch.tensor([8, 3], dtype=torch.int32, device="npu"),
    actual_seq_lengths_kv=torch.empty(2, dtype=torch.int32, device="npu"),
    hbm_block_table=torch.empty((2, 8), dtype=torch.int32, device="npu"),
    capacity=4,
)
```
