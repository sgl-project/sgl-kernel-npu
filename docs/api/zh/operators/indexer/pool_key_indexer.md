# pool_key_indexer

## 芯片支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `cann_ops_transformer` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import pool_key_indexer

def pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    topk=2048,
    pool_size=16,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
    q_descale=None,
    k_descale=None,
    *,
    impl="auto",
):
    ...
```

`auto` 选择 `cann_ops_transformer`，首次执行时加载随 wheel 打包的绑定。

## 功能说明

对 pooled key 打分并选择排名靠前的 pool，将其展开为 token 位置，再追加末尾尚未组成完整 pool 的 token。

原样返回 CANN 的 `(indices, values)` 元组。indices 为 int32，无效位置填 `-1`；`return_value=True` 时 values 为选中 pool 的 float32 分数，否则为空 Tensor。打分、掩码和选中 pool 的排列由 CANN 决定，不保证 pool 顺序稳定。输入不被修改。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype / 位置 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[B,S,H,D]` / `[T,H,D]` | CANN 支持的 query dtype | 必填 | BSND / TND query，原样转发。 |
| `pool_key` | `torch.Tensor` | `[B,P,1,D]` (BSND); layout-dependent for PA | 与 query 相同 | 必填 | pool key；PA 分页布局须配套页表。 |
| `weights` | `torch.Tensor` | `[B,S,H]` (BSND) / `[T,H]` (TND) | CANN 支持的 weight dtype | 必填 | 逐 query、逐 head 的 indexer 权重。 |
| `pool_tail_k` | `torch.Tensor` | `[B]` | CPU/NPU int64 | 必填 | 每请求尚未压缩的 tail token 数。 |
| `actual_seq_q` | `torch.Tensor` 或 None | `[B]` 或 — | CPU/NPU int64 | `None` | 打包 query 的累积结束位置。 |
| `actual_seq_k` | `torch.Tensor` 或 None | `[B]` 或 — | CPU/NPU int64 | `None` | TND 的 pooled key 累计结束位置，或 PA_BBND 的每请求 pooled key 数量。 |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | 分页 key 的 NPU 页 ID；None 转为空 optional tensor。 |
| `topk` | `int` | — | — | `2048` | token 级选取宽度，取值范围及 pool_size 整除约束由 CANN 校验。 |
| `pool_size` | `int` | — | — | `16` | 每个完整 pool 对应的 token 数，须为正数。 |
| `layout_q` | `str` | — | — | `"BSND"` | query 布局：BSND 或 TND。 |
| `layout_k` | `str` | — | — | `"BSND"` | key 布局为 BSND、TND 或 PA_BBND，支持范围由 CANN 校验。 |
| `mask_mode` | `int` | — | — | `3` | 原样透传 CANN。 |
| `quant_mode` | `int` | — | — | `-1` | `-1` 表示不量化，`0` 表示普通 FP8，`2` 表示 INT8。 |
| `q_descale` | `torch.Tensor` 或 `None` | `query.shape[:-1]` | 普通 FP8 用 FP32，INT8 用 FP16 | `None` | 每个 query token 和 head 一个反量化系数。普通 FP8 和 INT8 模式必传。 |
| `k_descale` | `torch.Tensor` 或 `None` | `pool_key.shape[:-1]` | 普通 FP8 用 FP32，INT8 用 FP16 | `None` | 每个 pooled-key 行和 head 一个反量化系数。普通 FP8 和 INT8 模式必传。 |
| `return_value` | `bool` | — | — | `False` | 透传 CANN，控制 values 是否包含选中 pool 的分数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字：auto、cann_ops_transformer 或 torch_native。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B,S,topk+K-1]`（BSND）或 `[T,topk+K-1]`（TND） | `torch.int32` | 元组第一项，位于 query 设备；K 为 pool_size。 |
| `values` | `torch.Tensor` | `[B,S,topk//K]`（BSND）、`[T,topk//K]`（TND）或 `[0]` | `torch.float32` | 元组第二项，return_value=False 时为空，无效分数填 -inf。 |

计算 tensor 位于 query 所在 NPU；长度元数据须为 CPU 或 NPU 上的 int64 tensor。
图回放时元数据须为 query NPU 上的连续 tensor，通过原地更新 buffer 修改；
全 CPU 元数据作为捕获时常量，修改后需重新捕获。
TND 空历史或全 padding 场景先调用 `prepare_pool_key_indexer_inputs`，
再按原始 query 边界对裁剪后的输出执行 packing。

## cann_ops_transformer 后端

要求 CANN 9.2.0。A3 由 sgl-kernel-npu wheel 随包提供 ops-transformer 绑定和 vendor；A5 加载系统 `cann_ops_transformer` 包。绑定原样转发参数并返回完整的 CANN 结果元组。

`auto`/`cann_ops_transformer` 为同一实现。

INT8 模式要求 query 和 pooled key 为 INT8，weights 和两个 descale 为 FP16。
在 `TND/PA_BBND` 布局下，`q_descale` 形状为 `[T, H]`，`k_descale` 形状为
`[block_num, block_size, N2]`，页的顺序与 `pool_key` 一致。INT8 的 Q/K 保持原坐标系。

## torch_native 后端

独立 FP32 reference，支持 BSND/BSND 和 TND/{TND, PA_BBND}，可在 CPU 或 NPU 执行。
返回相同的元组结构，并按 return_value 返回分数或空 Tensor。
普通 FP8 和 INT8 模式下，参考实现先以 FP32 将每个 query 和 pooled-key 向量乘以对应的 scale，再计算分数。

## 调用示例

示例使用 A3 和 CANN 9.2.0 wheel，演示 INT8 分页 key。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import pool_key_indexer

query = torch.zeros(4, 32, 128, dtype=torch.int8, device="npu")
query[..., 0] = 1
keys = torch.zeros(4, 16, 1, 128, dtype=torch.int8, device=query.device)
keys[..., 0] = torch.arange(1, 17, dtype=torch.int8, device=query.device).view(1, 16, 1)
weights = torch.ones(4, 32, dtype=torch.float16, device=query.device)
q_descale = torch.full((4, 32), 0.5, dtype=torch.float16, device=query.device)
k_descale = torch.full((4, 16, 1), 0.25, dtype=torch.float16, device=query.device)
indices, values = pool_key_indexer(
    query,
    keys,
    weights,
    pool_tail_k=torch.full((4,), 3, dtype=torch.int64, device=query.device),
    actual_seq_q=torch.arange(1, 5, dtype=torch.int64, device=query.device),
    actual_seq_k=torch.full((4,), 16, dtype=torch.int64, device=query.device),
    block_table=torch.arange(4, dtype=torch.int32, device=query.device).view(4, 1),
    topk=16,
    pool_size=4,
    layout_q="TND",
    layout_k="PA_BBND",
    quant_mode=2,
    q_descale=q_descale,
    k_descale=k_descale,
)
assert indices.shape == (4, 19) and indices.dtype == torch.int32
assert values.shape == (0,)
```
