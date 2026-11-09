# mixed_quant_sparse_flash_mla_metadata

## 功能与支持范围

MQSMLA 的 AICPU 调度算子：根据两侧 TopK 长度生成整行或 FlashDecode 分片执行计划。

DSL 面向 Ascend 950PR/950DT（`arch35`），设备验证状态见下表，`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata


def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
    impl="auto"
): ...
```

## 输入约束

`ori_topk_length` 与 `cmp_topk_length` 是每个 Query 行两侧的有效稀疏长度；`has_ori_kv` 与 `has_cmp_kv` 应与主算子提供的 KV 侧一致。`num_heads_q=64`、`num_heads_kv=1`、`head_dim=512`、`quant_mode=1`。

## 返回值与调用顺序

返回 INT32 `[1024]` metadata；同一 NPU stream 中先生成，再传给 `mixed_quant_sparse_flash_mla(..., metadata=metadata)`。

## 硬件支持

| impl | Ascend 950PR | Ascend 950DT | Ascend 910C |
| --- | --- | --- | --- |
| cannbot_dsl | 支持 | 未验证 | 不支持 |

## 参数

T 为查询行数，B 为请求数，P 为物理页数，S 为每页槽位数。可选张量默认 None；其他默认值见函数原型。

| 参数 | 类型、形状与含义 |
| --- | --- |
| `ori_topk_length` | INT32 [T,1]，每个查询原始稀疏索引的有效前缀长度；启用原始 KV 时必需。 |
| `cmp_topk_length` | INT32 [T,1]，每个查询压缩稀疏索引的有效前缀长度；启用压缩 KV 时必需。 |
| `cu_seqlens_q` | 可选 INT32 [B+1]，各请求查询行的累计边界。 |
| `seqused_q` | 可选 INT32 [B]，每个请求的有效查询行数。 |
| `seqused_ori_kv` | 可选 INT32 [B]，原始 KV 长度；保留该兼容参数。 |
| `seqused_cmp_kv` | 可选 INT32 [B]，压缩 KV 长度；保留该兼容参数。 |
| `batch_size` | 可选 int B，请求数；省略时从查询边界推导。 |
| `max_seqlen_q` | int 或 None，每个请求的查询容量；未指定时推导。 |
| `max_seqlen_ori_kv` | 可选 int，原始 KV 容量（兼容参数）。 |
| `max_seqlen_cmp_kv` | 可选 int，压缩 KV 容量（兼容参数）。 |
| `num_heads_q` | int，查询头数：Indexer 为 32，MLA 为 64。 |
| `num_heads_kv` | int，共享 MLA KV 头数，使用 1。 |
| `head_dim` | int，逻辑头维度：Indexer 为 128，MLA 为 512。 |
| `quant_mode` | int，当前支持的量化配置使用 1。 |
| `layout_q` | str，使用 TND（展平查询 token）。 |
| `layout_kv` | str，使用 PA_BBND 打包分页 KV。 |
| `has_ori_kv` | bool，消费该 metadata 的 attention 是否包含原始 KV。 |
| `has_cmp_kv` | bool，消费该 metadata 的 attention 是否包含压缩 KV。 |
| `impl` | str，auto 与 cannbot_dsl 选择同一个 AICPU 调度实现。 |

## 返回值

| 值 | 含义 |
| --- | --- |
| metadata | INT32 [1024]，后端内部调度格式。 |

## cannbot_dsl

`auto` 默认选择此后端。需要 A5、cannbot-dsl 0.8.0；本项目验证目标为 CANN 9.2.0。输入在同一设备，首次调用 JIT 编译；图捕获前预热实际形状。

metadata 是该设备实现的调度计划，不存在唯一的 Torch 数值答案，因此不提供同名 torch_native 别名。配套计算算子的独立参考验证结果语义，本入口测试长度更新后的 eager/graph 一致性。

## 示例

以下代码构造一组输入并生成 metadata：

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

q = torch.randn(1, 64, 512, device="npu", dtype=torch.bfloat16)
kv = torch.zeros(1, 128, 1, 544, device="npu", dtype=torch.uint8)
kv[..., 512:] = torch.ones(1, 128, 1, 16, device="npu", dtype=torch.bfloat16).view(
    torch.uint8
)
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
length = torch.tensor([[128]], device="npu", dtype=torch.int32)
metadata = mixed_quant_sparse_flash_mla_metadata(
    length,
    None,
    cu_seqlens_q=cu,
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    quant_mode=1,
    has_cmp_kv=False,
)
```

## 设备测试

```bash
python -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py --npu-device=0 --enable-perf=false -v -s
```
