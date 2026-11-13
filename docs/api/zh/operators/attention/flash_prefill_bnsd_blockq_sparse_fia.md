# flash_prefill_bnsd_blockq_sparse_fia

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention import flash_prefill_bnsd_blockq_sparse_fia


def flash_prefill_bnsd_blockq_sparse_fia(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    topk_idx,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    sm_scale,
    num_pages,
    max_num_blocks,
    block_table_out=None,
    actual_kvlen_out=None,
    *,
    impl="auto"
): ...
```

## 功能描述

从选定逻辑 KV 块计算稀疏因果 prefill 注意力：收集物理页，把当前块放在历史块之后，按当前查询的因果长度截断，再计算分组查询注意力。仅支持一个 KV 头，可选元数据缓冲区会原地更新。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `q` | Tensor | `Required` | KDA 为 [1,tokens,Hq,K]；FIA 为 FP16/BF16 [tokens,QH,D]。 |
| `k_cache_bnsd` | Tensor | `Required` | FP16/BF16 缓存 [num_pages,block_size,1,D]。 |
| `v_cache_bnsd` | Tensor | `Required` | 与 key 同形状、同类型的 value 缓存。 |
| `topk_idx` | Tensor | `Required` | INT32 选中逻辑块 [1,tokens,topk+1]，-1 补齐，包含当前块。 |
| `seq_lens` | Tensor | `Required` | 整数请求查询长度；FIA 接口中表示逐查询的因果 KV 长度。 |
| `per_query_req` | Tensor | `Required` | INT32 请求索引 [tokens]。 |
| `req_to_token` | Tensor | `Required` | INT32 物理 token 槽 [requests,max_context]。 |
| `block_size` | int | `Required` | 每个物理 KV 页的 token 数。 |
| `sm_scale` | float or None | `Required` | 注意力分数缩放，None 使用 head_dim**-0.5。 |
| `num_pages` | int | `Required` | 物理缓存页数，与缓存形状一致。 |
| `max_num_blocks` | int | `Required` | 保留的最大选块数量。 |
| `block_table_out` | Tensor or None | `None` | 可选可写 INT32 页表 [tokens,topk+1]。 |
| `actual_kvlen_out` | Tensor or None | `None` | 可选可写 INT32 因果长度 [tokens]。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | [tokens,QH,D] 输出，与 q 同类型、同设备。 |

## triton

使用 Triton 页表准备及原生 NPU 融合注意力接口。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_prefill_bnsd_blockq_sparse_fia

q = torch.randn(1, 16, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(2, 128, 1, 128, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
blocks = torch.tensor([[[-1, 0]]], device="npu", dtype=torch.int32)
lengths = torch.tensor([4], device="npu", dtype=torch.int32)
request = torch.tensor([0], device="npu", dtype=torch.int32)
mapping = torch.arange(256, device="npu", dtype=torch.int32).view(1, -1)
y = flash_prefill_bnsd_blockq_sparse_fia(
    q, k, v, blocks, lengths, request, mapping, 128, None, 2, 2, impl="triton"
)
```
