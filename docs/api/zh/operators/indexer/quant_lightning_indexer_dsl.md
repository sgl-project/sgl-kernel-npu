# quant_lightning_indexer_dsl

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | 不支持 | 支持 |
| torch_native | 不支持 | 支持 |

## 函数原型

`from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_dsl`

```python
import torch
from typing import Optional

def quant_lightning_indexer_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    impl="auto"
): ...
```

## 功能说明

MXFP4 Lightning Indexer：遍历可见 Key，按 Query head 加权得分选取 TopK token；可同时输出 QSLI 的候选块。

`q` 为 UINT8 `[T1,32,64]`，`k` 为打包 MXFP4 Key（PA_BBND 或 TND），`descale_q` 与 `descale_k` 为 E8M0 scale。`metadata` 必须由同配置的 `quant_lightning_indexer_metadata_dsl` 生成。`quant_mode=1`；`layout_q="TND"`，`layout_k` 支持 `TND`/`PA_BBND`；候选输出启用时 `candidate_block_size=8`。

计算语义（量化边界使用 BF16）：

```text
s_i = sum_h w_h * max(dot(dequant(q_h), dequant(k_i)), 0); indices = TopK(s)
```

`dequant(code, e) = signed_E2M1(code) * 2**(e - 127)`，每 32 个逻辑值共用一个 E8M0 指数字节，无 zero point。E2M1 幅值为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，每字节低半字节存第一个值。QK 结果、头权重及每次加权头累加的结果均舍入到 BF16。

## 参数

T 为查询行数，B 为请求数，P 为物理页数，S 为每页槽位数。

| 参数 | 默认值 | 类型、形状与含义 |
| --- | --- | --- |
| `q` | 必填 | UINT8 [T,32,64]，逻辑维度 128 的 MXFP4 查询。 |
| `k` | 必填 | UINT8 K：TND 为 [K,1,64]，PA_BBND 为 [P,S,1,64]；QSLI PA 使用合并的 [P,S/8,544] 字节。 |
| `w` | 必填 | FP32 [T,32]，查询头的分数权重。 |
| `descale_q` | 必填 | UINT8/E8M0 [T,32,2,2]，每 32 个逻辑特征一个 scale。 |
| `descale_k` | 必填 | UINT8/E8M0 [K,1,2,2] 或 [P,S,1,2,2]；QSLI 合并 PA 存储时省略。 |
| `cu_seqlens_q` | `None` | 可选 INT32 [B+1]，各请求查询行的累计边界。 |
| `cu_seqlens_k` | `None` | 可选 INT32 [B+1]，TND K 的累计边界。 |
| `seqused_q` | `None` | 可选 INT32 [B]，每个请求的有效查询行数。 |
| `seqused_k` | `None` | 可选 INT32 [B]，每个请求的有效 K token 数。 |
| `cmp_residual_k` | `None` | 可选 INT32 [B]，因果压缩未成组的残余 token 数。 |
| `block_table` | `None` | 可选 INT32 [B,max_blocks]，PA 逻辑页到物理页的映射。 |
| `output_idx_offset` | `None` | 可选 INT32 [T,1]，加到有效返回 token 索引上的偏移。 |
| `metadata` | `None` | cannbot_dsl 必需的 INT32 [1024] 设备调度数据；torch_native 不使用该参数。 |
| `topk` | 必填 | int，每个查询最多选择的 token 数，范围 [1,8192]。 |
| `quant_mode` | 必填 | int，当前支持的量化配置使用 1。 |
| `max_seqlen_q` | `-1` | int 或 None，每个请求的查询容量；未指定时推导。 |
| `mask_mode` | `0` | int：0 使用可见 K，3 使用因果掩码。 |
| `cmp_ratio` | `1` | int，范围 [1,128]，用于计算因果可见长度的压缩率。 |
| `layout_q` | `'TND'` | str，使用 TND（展平查询 token）。 |
| `layout_k` | `'TND'` | str，TND 或 PA_BBND 的 K 存储布局。 |
| `return_value` | `False` | bool，True 时同时返回 BF16 选中分数。 |
| `candidate_topk_blocks` | `-1` | int，候选块输出容量；-1 禁用该输出。 |
| `candidate_block_size` | `-1` | int，启用候选块时每块使用 8 个 token。 |
| `impl` | `'auto'` | str，auto 选择 cannbot_dsl；torch_native 是独立计算参考。 |

## 返回值

| 值 | 含义 |
| --- | --- |
| indices | INT32 [T,1,topk]; -1 表示无效项。 |
| values | BF16 [T,1,topk]; 禁用时为空，无效项为 -inf。 |
| candidates, candidate_length | INT32 [T,1,candidate_topk_blocks], INT32 [T,1]; 禁用时为空。 |

## cannbot_dsl

`auto` 默认选择此后端。需要 A5 和环境中安装的兼容 CANNBot-DSL 运行时，算子入口不强制检查版本。输入在同一设备，首次调用 JIT 编译；图捕获前预热实际形状。

## torch_native

独立的 CPU Torch 精度参考，结果复制回输入设备，保留原地输出语义。无需 DSL；不适合性能执行或 NPU graph capture。Top-K 同分项允许不同排序。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import (
    quant_lightning_indexer_dsl,
    quant_lightning_indexer_metadata_dsl,
)

q = torch.zeros(1, 32, 64, device="npu", dtype=torch.uint8)
qs = torch.full((1, 32, 2, 2), 127, device="npu", dtype=torch.uint8)
w = torch.ones(1, 32, device="npu")
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
lengths = torch.tensor([64], device="npu", dtype=torch.int32)
table = torch.zeros(1, 1, device="npu", dtype=torch.int32)
attrs = dict(topk=512, layout_q="TND", layout_k="PA_BBND")
k = torch.zeros(1, 64, 1, 64, device="npu", dtype=torch.uint8)
ks = torch.full((1, 64, 1, 2, 2), 127, device="npu", dtype=torch.uint8)
metadata = quant_lightning_indexer_metadata_dsl(
    cu_seqlens_q=cu,
    seqused_k=lengths,
    num_heads_q=32,
    num_heads_k=1,
    head_dim=128,
    max_seqlen_q=1,
    max_seqlen_k=64,
    **attrs
)
result = quant_lightning_indexer_dsl(
    q,
    k,
    w,
    qs,
    ks,
    cu_seqlens_q=cu,
    seqused_k=lengths,
    block_table=table,
    metadata=metadata,
    quant_mode=1,
    **attrs
)
```

## 设备测试

```bash
python -m pytest tests/python/operators/indexer/test_quant_lightning_indexer_dsl.py --npu-device=0 --enable-perf=false -v -s
```
