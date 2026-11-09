# quant_sparse_lightning_indexer

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | 不支持 | 支持 |
| torch_native | 不支持 | 支持 |

## 函数原型

`from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer`

```python
import torch
from typing import Optional

def quant_sparse_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    descale_k: Optional[torch.Tensor] = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    impl="auto"
): ...
```

## 功能说明

MXFP4 Sparse Lightning Indexer：只访问 QLI 给出的候选 Key 块，再选出 TopK token。

`q` 为 UINT8 `[T1,32,64]`；`candidate_block_indices` 为 INT32 候选块号，`candidate_block_length` 给出每行有效前缀。`descale_k` 为 E8M0 scale，`metadata` 需由同配置的 QSLI metadata 算子生成。`candidate_block_size=8`、`quant_mode=1`；K 可为 `TND` 或 `PA_BBND`。

计算语义（量化边界使用 BF16）：

```text
s_i = sum_h w_h * max(dot(dequant(q_h), dequant(k_i)), 0), i in visible candidate blocks; indices = TopK(s)
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
| `candidate_block_indices` | 必填 | INT32 [T,1,2048]，候选逻辑块 ID；有效前缀由 length 指定。 |
| `candidate_block_length` | 必填 | INT32 [T,1]，每个查询的有效候选块数。 |
| `cu_seqlens_q` | `None` | 可选 INT32 [B+1]，各请求查询行的累计边界。 |
| `cu_seqlens_k` | `None` | 可选 INT32 [B+1]，TND K 的累计边界。 |
| `seqused_q` | `None` | 可选 INT32 [B]，每个请求的有效查询行数。 |
| `seqused_k` | `None` | 可选 INT32 [B]，每个请求的有效 K token 数。 |
| `cmp_residual_k` | `None` | 可选 INT32 [B]，因果压缩未成组的残余 token 数。 |
| `block_table` | `None` | 可选 INT32 [B,max_blocks]，PA 逻辑页到物理页的映射。 |
| `output_idx_offset` | `None` | 可选 INT32 [T,1]，加到有效返回 token 索引上的偏移。 |
| `metadata` | `None` | cannbot_dsl 必需的 INT32 [1024] 设备调度数据；torch_native 不使用该参数。 |
| `descale_k` | `None` | UINT8/E8M0 [K,1,2,2] 或 [P,S,1,2,2]；QSLI 合并 PA 存储时省略。 |
| `topk` | 必填 | int，每个查询最多选择的 token 数，范围 [1,8192]。 |
| `candidate_block_size` | 必填 | int，启用候选块时每块使用 8 个 token。 |
| `quant_mode` | 必填 | int，当前支持的量化配置使用 1。 |
| `max_seqlen_q` | `-1` | int 或 None，每个请求的查询容量；未指定时推导。 |
| `mask_mode` | `0` | int：0 使用可见 K，3 使用因果掩码。 |
| `cmp_ratio` | `1` | int，范围 [1,128]，用于计算因果可见长度的压缩率。 |
| `layout_q` | `'TND'` | str，使用 TND（展平查询 token）。 |
| `layout_k` | `'TND'` | str，TND 或 PA_BBND 的 K 存储布局。 |
| `return_value` | `False` | bool，True 时同时返回 BF16 选中分数。 |
| `impl` | `'auto'` | str，auto 选择 cannbot_dsl；torch_native 是独立计算参考。 |

## 返回值

| 值 | 含义 |
| --- | --- |
| indices | INT32 [T,1,topk]; -1 表示无效项。 |
| values | BF16 [T,1,topk]; 禁用时为空，无效项为 -inf。 |

## cannbot_dsl

`auto` 默认选择此后端。需要 A5 和环境中安装的兼容 CANNBot-DSL 运行时，算子入口不强制检查版本。输入在同一设备，首次调用 JIT 编译；图捕获前预热实际形状。

## torch_native

独立的 CPU Torch 精度参考，结果复制回输入设备，保留原地输出语义。无需 DSL；不适合性能执行或 NPU graph capture。Top-K 同分项允许不同排序。

## 示例

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
result = quant_sparse_lightning_indexer(
    q,
    k,
    w,
    qs,
    candidates,
    count,
    cu_seqlens_q=cu,
    seqused_k=lengths,
    block_table=table,
    metadata=metadata,
    quant_mode=1,
    **attrs,
    candidate_block_size=8
)
```

## 设备测试

```bash
python -m pytest tests/python/operators/indexer/test_quant_sparse_lightning_indexer.py --npu-device=0 --enable-perf=false -v -s
```
