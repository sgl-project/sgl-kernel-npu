# prepare_sparse_kv_offload_rows

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| triton | 未验证 | 未验证 |
| torch_native | 未验证 | 未验证 |

## 接口

`auto` 默认选择 Triton。

```python
prepare_sparse_kv_offload_rows(tokens, device, request_ids, seq_lens, positions, out_cache_loc, *, request_rows, context_len, host_capacity, query_width=1, extend_seq_lens=None, extend_prefix_lens=None, compact_prefill=False, impl="auto")
```

## 功能

为稀疏 KV offload 准备源行索引、host 目标索引、token 位置和有效标记。Decode 按 `query_width` 个 token 映射一个请求；prefill 支持变长紧凑布局和等宽填充布局。提供 `extend_seq_lens` 时进入 prefill，缺失的 prefix 条目按零处理。

## 参数和返回值

| 参数 | 含义 |
| --- | --- |
| `tokens`, `device` | 输出长度（int）和 torch 设备 |
| `request_ids`, `seq_lens` | 长度为 B 的一维整数张量，分别为请求 ID 和序列长度 |
| `positions`, `out_cache_loc` | 长度为 tokens 的一维整数张量，分别为 decode 位置和目标索引；prefill 时 positions 可为 None |
| `request_rows`, `context_len`, `host_capacity` | int 类型的请求、位置和目标索引上界 |
| `query_width` | 每个请求的 decode token 数，默认为 1 |
| `extend_seq_lens`, `extend_prefix_lens` | 一维整数张量，分别是 prefill 长度和前缀长度。Decode 时默认为 None。提供 extend 长度时也须传入前缀张量，可以为空。 |
| `compact_prefill` | 是否使用紧凑 prefill 布局，默认为 False |
| `impl` | 后端：auto、triton 或 torch_native |

依次返回四个形状为 [tokens] 的张量：源索引、目标索引和 token 位置为 int64，有效标记为 bool。填充 prefill 时 tokens 必须能被 B 整除。输入须与输出位于同一设备。

## triton 后端

使用 Triton kernel。

## torch_native 后端

使用 PyTorch 操作作为独立参考实现。

## 示例

```python
import torch
from sgl_kernel_npu.operators.mem_cache import prepare_sparse_kv_offload_rows

device = torch.device("npu")
request_ids = torch.tensor([1], dtype=torch.int32, device=device)
seq_lens = torch.tensor([7], dtype=torch.int32, device=device)
query_positions = torch.tensor([6], dtype=torch.int32, device=device)
out_cache_loc = torch.tensor([12], dtype=torch.int32, device=device)
source, destination, positions, valid = prepare_sparse_kv_offload_rows(
    1, device, request_ids, seq_lens, query_positions, out_cache_loc,
    request_rows=2, context_len=32, host_capacity=64, impl="auto",
)
# [0], [12], [6], [True]
```
