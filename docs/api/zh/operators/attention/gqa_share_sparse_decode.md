# gqa_share_sparse_decode

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode

def gqa_share_sparse_decode(q, sink, k_cache_bnsd, v_cache_bnsd, block_table, seq_lens, block_size, topk_idx, sm_scale=None, num_topk_chunks=None, max_num_topk_chunks=8, req_to_token=None, req_pool_indices=None, max_num_blocks=None, num_pages=None, sanitize_page_ids=False, topk_blocks_per_step=1, prefetch_idx=False, num_warps=None, num_stages=None, *, impl='auto'):
    ...
```

默认 `auto` 与 `triton` 注册到同一入口；不自动回退到参考实现。

## 功能说明

收集选中 KV 页，计算 $softmax(QK^\mathsf{T}\,sm\_scale)V$，屏蔽超出 seq_lens 的 token。可选 sink 提供值为零的额外项。

## 参数与返回值

Triton 张量输入应位于同一 NPU 并采用连续布局；输出位于输入设备。

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `q` | torch.Tensor | `[B,QH,D]; float16/bfloat16` | 必填 | 解码查询。 |
| `sink` | torch.Tensor 或 None | `[QH,D]; q.dtype` | 必填 | 参与 softmax 的额外键，其值为零。 |
| `k_cache_bnsd` | torch.Tensor | `[pages,block_size,KVH,D]; q.dtype` | 必填 | 分页键缓存。 |
| `v_cache_bnsd` | torch.Tensor | `[pages,block_size,KVH,D]; q.dtype` | 必填 | 分页值缓存，形状与键缓存相同。 |
| `block_table` | torch.Tensor 或 None | `[B,max_blocks]; int32/int64` | 必填 | 逻辑页到物理页映射；直接查表模式传 None。 |
| `seq_lens` | torch.Tensor | `[T] or [B]; int32` | 必填 | 每个查询或请求的因果 KV 长度。 |
| `block_size` | int | `—` | 必填 | 每个 KV 页的 token 数；须为正。 |
| `topk_idx` | torch.Tensor | `[KVH or QH,B,S]; int32` | 必填 | 选中的逻辑块，-1 填充；QH 布局取每组首个头。 |
| `sm_scale` | float 或 None | `—` | `None` | 注意力缩放；None 使用 D**-0.5。 |
| `num_topk_chunks` | int 或 None | `—` | `None` | 二次幂分块数，不超过 max(1,S)；None 自动选择。 |
| `max_num_topk_chunks` | int | `—` | `8` | 自动选择分块数的上限。 |
| `req_to_token` | torch.Tensor 或 None | `[requests,max_ctx]; int32/int64` | `None` | token 槽映射；直接查表模式须提供 req_pool_indices。 |
| `req_pool_indices` | torch.Tensor 或 None | `[B]; int32/int64` | `None` | 直接查表模式的请求索引。 |
| `max_num_blocks` | int 或 None | `—` | `None` | 正的逻辑块容量，直接查表模式必填。 |
| `num_pages` | int 或 None | `—` | `None` | 正的物理页数，直接查表模式必填。 |
| `sanitize_page_ids` | bool | `—` | `False` | 将直接查表的物理页号限制在有效范围内。 |
| `topk_blocks_per_step` | int | `—` | `1` | 每个计算块处理的选中块数，须为二次幂。 |
| `prefetch_idx` | bool | `—` | `False` | 每步处理单块时预取索引。 |
| `num_warps` | int 或 None | `—` | `None` | Triton 启动参数；None 使用后端默认值。 |
| `num_stages` | int 或 None | `—` | `None` | Triton 流水参数；None 使用后端默认值。 |
| `impl` | str | `—` | `'auto'` | 仅限关键字：auto、triton 或 torch_native。 |

| 返回值 | 类型 | 形状 / 数据类型 | 描述 |
| --- | --- | --- | --- |
| `output` | torch.Tensor 或 None | `q.shape; q.dtype` | 新建的稀疏注意力输出；不修改输入。 |

## triton 后端

q/K/V 须位于同一 NPU 且 FP16/BF16 类型相同，QH 须能被 KVH 整除，页映射须有效。公开 Triton 入口关闭可选原生注意力路径。

## torch_native 后端

通过独立 PyTorch 运算生成参考结果；主机元数据传输可能触发同步，仅用于精度对比。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode

q = torch.randn(1, 4, 64, device="npu", dtype=torch.float16)
k = torch.randn(4, 16, 2, 64, device="npu", dtype=torch.float16)
v = torch.randn_like(k)
table = torch.tensor([[0, 1, 2, 3]], device="npu", dtype=torch.int32)
lengths = torch.tensor([32], device="npu", dtype=torch.int32)
topk = torch.tensor([[[0, 1]], [[0, 1]]], device="npu", dtype=torch.int32)
y = gqa_share_sparse_decode(q, None, k, v, table, lengths, 16, topk, impl="triton")
```
