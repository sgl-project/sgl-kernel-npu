# quant_sparse_lightning_indexer_metadata

## 芯片支持

| impl | Ascend 950PR | Ascend 950DT | Ascend 910C |
| --- | --- | --- | --- |
| cannbot_dsl | 支持 | 未验证 | 不支持 |

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer_metadata


def quant_sparse_lightning_indexer_metadata(
    candidate_block_length,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    candidate_block_size,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    impl="auto"
): ...
```

## 功能说明

QSLI 的 AICPU 调度算子：按每行候选长度和序列可见范围规划分核与跨核归并。

DSL 面向 Ascend 950PR/950DT（`arch35`），设备验证状态见芯片支持表，`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。

`candidate_block_length` 为每个 Query 的有效候选块数量，`cu_seqlens_q/k`、`seqused_q/k` 描述存储和有效长度；`candidate_block_size=8`、`quant_mode=1` 应与主算子一致。候选长度或静态属性变化后需重建 metadata；不可用 QLI metadata 替代。

## 参数

T 为查询行数，B 为请求数，P 为物理页数，S 为每页槽位数。

| 参数 | 默认值 | 类型、形状与含义 |
| --- | --- | --- |
| `candidate_block_length` | 必填 | INT32 [T,1]，每个查询的有效候选块数。 |
| `cu_seqlens_q` | `None` | 可选 INT32 [B+1]，各请求查询行的累计边界。 |
| `cu_seqlens_k` | `None` | 可选 INT32 [B+1]，TND K 的累计边界。 |
| `seqused_q` | `None` | 可选 INT32 [B]，每个请求的有效查询行数。 |
| `seqused_k` | `None` | 可选 INT32 [B]，每个请求的有效 K token 数。 |
| `cmp_residual_k` | `None` | 可选 INT32 [B]，因果压缩未成组的残余 token 数。 |
| `batch_size` | `None` | 可选 int B，请求数；省略时从查询边界推导。 |
| `max_seqlen_q` | `-1` | int 或 None，每个请求的查询容量；未指定时推导。 |
| `max_seqlen_k` | `-1` | int，每个请求的 K 容量；-1 表示推导。 |
| `num_heads_q` | 必填 | int，查询头数：Indexer 为 32，MLA 为 64。 |
| `num_heads_k` | 必填 | int，共享 Index K 头数，使用 1。 |
| `head_dim` | 必填 | int，逻辑头维度：Indexer 为 128，MLA 为 512。 |
| `topk` | 必填 | int，每个查询最多选择的 token 数，范围 [1,8192]。 |
| `quant_mode` | 必填 | int，当前支持的量化配置使用 1。 |
| `candidate_block_size` | 必填 | int，启用候选块时每块使用 8 个 token。 |
| `mask_mode` | `0` | int：0 使用可见 K，3 使用因果掩码。 |
| `cmp_ratio` | `1` | int，范围 [1,128]，用于计算因果可见长度的压缩率。 |
| `layout_q` | `'TND'` | str，使用 TND（展平查询 token）。 |
| `layout_k` | `'TND'` | str，TND 或 PA_BBND 的 K 存储布局。 |
| `impl` | `'auto'` | str，auto 与 cannbot_dsl 选择同一个 AICPU 调度实现。 |

## 返回值

| 值 | 含义 |
| --- | --- |
| metadata | INT32 [1024]，后端内部调度格式。 |

## cannbot_dsl

`auto` 默认选择此后端。需要 A5 和环境中安装的兼容 CANNBot-DSL 运行时，算子入口不强制检查版本。输入在同一设备，首次调用 JIT 编译；图捕获前预热实际形状。

metadata 是该设备实现的调度计划，不存在唯一的 Torch 数值答案，因此不提供同名 torch_native 别名。配套计算算子的独立参考验证结果语义，本入口测试长度更新后的 eager/graph 一致性。

## 示例

以下代码构造一组输入并生成 metadata：

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import (
    quant_sparse_lightning_indexer,
    quant_sparse_lightning_indexer_metadata,
)

q = torch.zeros(1, 32, 64, device="npu", dtype=torch.uint8)
qs = torch.full((1, 32, 2, 2), 127, device="npu", dtype=torch.uint8)
w = torch.ones(1, 32, device="npu")
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
lengths = torch.tensor([64], device="npu", dtype=torch.int32)
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
attrs = dict(topk=512, layout_q="TND", layout_k="PA_BBND")
k = torch.zeros(1, 8, 544, device="npu", dtype=torch.uint8)
k[..., 512:] = 127
candidates = torch.full((1, 1, 2048), -1, device="npu", dtype=torch.int32)
candidates[..., :8] = torch.arange(8, device="npu", dtype=torch.int32)
count = torch.tensor([[8]], device="npu", dtype=torch.int32)
metadata = quant_sparse_lightning_indexer_metadata(
    cu_seqlens_q=cu,
    seqused_k=lengths,
    num_heads_q=32,
    num_heads_k=1,
    head_dim=128,
    max_seqlen_q=1,
    max_seqlen_k=64,
    **attrs,
    candidate_block_length=count,
    candidate_block_size=8,
    quant_mode=1
)
```

## 设备测试

```bash
python -m pytest tests/python/operators/indexer/test_quant_sparse_lightning_indexer_metadata.py --npu-device=0 --enable-perf=false -v -s
```
