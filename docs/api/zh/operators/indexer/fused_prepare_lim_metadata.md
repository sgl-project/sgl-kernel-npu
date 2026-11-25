# fused_prepare_lim_metadata

## 硬件支持

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| triton | 未验证 | 未验证 |
| torch_native | 未验证 | 未验证 |

## 接口

`auto` 选择 Triton。

```python
from sgl_kernel_npu.operators.indexer import fused_prepare_lim_metadata

def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity: int,
    request_rows: int,
    query_width: int = 1,
    positions=None,
    resident_safe_slots=None,
    request_to_resident=None,
    resident_rows=None,
    impl="auto",
) -> None:
    ...
```

## 说明

原地准备 LIM 请求元数据和页表；无效请求使用哨兵，稀疏状态由容量及历史前缀决定。

## 参数及返回值

| 参数 | 形状、类型和含义 |
| --- | --- |
| `seq_lens` | [B] int32，活动长度 |
| `request_slots` | [B] int32，请求槽位 |
| `last_prefix` | [resident_rows + 1] int32，历史前缀，包含哨兵 |
| `block_table` | [B, S] int32，源页表，可使用任意步幅 |
| `out_block_table` | [B, D] int32，输出页表，可使用任意步幅 |
| `safe_slots` | [B] int32，输出请求槽位或哨兵 |
| `lengths` | [B] int32，输出可见长度 |
| `sparse_rows` | [B] bool，稀疏标记 |
| `valid_rows` | [B] bool，有效标记 |
| `request_states` | [B] int32，输出状态 |
| `next_prefixes` | [B] int32，输出下次前缀 |
| `cache_sizes` | [B] int32，输出缓存大小 |
| `capacity` | int，缓存容量 |
| `request_rows` | int，请求槽位数，不含哨兵 |
| `query_width` | int，每请求查询数，默认 1 |
| `positions` | 可选的连续整数张量，包含 B × query_width 个查询位置，默认 None |
| `resident_safe_slots` | 可选 [B] int32，输出驻留槽位，默认 None |
| `request_to_resident` | 可选 [request_rows] int32，请求到驻留槽位映射，默认 None |
| `resident_rows` | 可选 int，驻留槽位数，不含哨兵，默认 None |
| `impl` | str，默认 auto，可选 triton、torch_native |

返回 `None`；输出张量原地修改。除注明外张量应连续并位于同一设备。

有效请求要求请求槽位和映射后的驻留槽位均大于零且小于各自行数，活动长度和可见长度均大于零。无效行输出哨兵槽位和零长度。positions 存在时可见长度为每行最后位置加一，否则取 seq_lens。长度达到 capacity 时为稀疏行：历史前缀非负且不大于当前长度时状态为 -1，否则为 -2；非稀疏状态为 -3。稀疏行下次前缀为长度、缓存大小为容量，其余分别为 -1 和 0。页表复制前 min(S, D) 列，其余置零。resident_rows 默认为 request_rows。使用映射时应同时提供 resident_safe_slots；否则两个输出共用同一存储，最终为驻留槽位。B、D、query_width 必须为正。

## triton backend

使用 Triton 内核。

## torch_native backend

使用 PyTorch 操作作为参考实现。

## 示例

```python
import torch
from sgl_kernel_npu.operators.indexer import fused_prepare_lim_metadata

def ints(values):
    return torch.tensor(values, device="npu", dtype=torch.int32)

visible = torch.empty_like(ints([4096]))
state = torch.empty_like(visible)
fused_prepare_lim_metadata(
    seq_lens=ints([4096]),
    request_slots=ints([1]),
    last_prefix=ints([-1, 4095, -1]),
    block_table=ints([list(range(32))]),
    out_block_table=ints([[0] * 32]),
    safe_slots=torch.empty_like(visible),
    lengths=visible,
    sparse_rows=torch.empty(1, device="npu", dtype=torch.bool),
    valid_rows=torch.empty(1, device="npu", dtype=torch.bool),
    request_states=state,
    next_prefixes=torch.empty_like(visible),
    cache_sizes=torch.empty_like(visible),
    capacity=4096,
    request_rows=2,
    impl="auto",
)
# visible contains 4096; state contains -1.
```
