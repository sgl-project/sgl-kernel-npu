# pool_key_indexer

## 芯片支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `cann_ops_transformer` | 支持 | 支持 |

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
    key_stride0=-1,
    *,
    impl="auto",
):
    ...
```

`auto` 固定选择 `cann_ops_transformer`，首次执行时延迟加载随包提供的 CANN 绑定；不进行后端回退。

## 功能说明

对 pooled key 打分并选择排名靠前的 pool，将其展开为 token 位置，再追加末尾尚未组成完整 pool 的 token。

输出为 int32 索引，不返回分数；未使用的 tail 位置填为 `-1`。打分、掩码和选中 pool 的排列由 vendor 算子决定，不保证 pool 顺序稳定。输入不被修改。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype / 位置 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[B,S,H,D]` / `[T,H,D]` / `[T,H*D]` | `float16` / `bfloat16` | 必填 | BSND / TND query；二维输入直接转发，不在 binding 中 reshape。 |
| `pool_key` | `torch.Tensor` | `[B,P,1,D]` (BSND); layout-dependent for PA | 与 query 相同 | 必填 | pool key；PA 分页布局须配套页表。 |
| `weights` | `torch.Tensor` | `[B,S,H]` (BSND) / `[T,H]` (TND) | 与 query 相同 | 必填 | 逐 query、逐 head 的 indexer 权重。 |
| `pool_tail_k` | `Sequence[int]` 或 Tensor | `[B]` | host 整数或 NPU int64 | 必填 | 每请求尚未压缩的 tail token 数。 |
| `actual_seq_q` | `Sequence[int]`、Tensor 或 None | `[B]` 或 — | host 整数或 NPU int64 | `None` | 打包 query 的累积结束位置。 |
| `actual_seq_k` | `Sequence[int]`、Tensor 或 None | `[B]` 或 — | host 整数或 NPU int64 | `None` | 每请求的完整压缩池数量。 |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | 分页 key 的 NPU 页 ID；None 转为空 optional tensor。 |
| `topk` | `int` | — | — | `2048` | token 级选取宽度，至少为 pool_size；按 pool_size 的倍数使用，并提供足够有效 pool。 |
| `pool_size` | `int` | — | — | `16` | 每个完整 pool 对应的 token 数，须为正数。 |
| `layout_q` | `str` | — | — | `"BSND"` | query 布局；示例涉及 BSND，测试另含 TND。 |
| `layout_k` | `str` | — | — | `"BSND"` | key 布局；测试配置使用 BSND 或 PA_BBND。 |
| `mask_mode` | `int` | — | — | `3` | 透传 vendor mask mode；当前接入使用模式 3。 |
| `quant_mode` | `int` | — | — | `-1` | 仅支持 -1；不对外暴露 descale tensor。 |
| `return_value` | `bool` | — | — | `False` | 必须为 False：schema 仅返回索引。 |
| `key_stride0` | `int` | — | — | `-1` | 兼容参数；CANN 从 `pool_key` 推导 stride，仅接受 -1 或 `pool_key.stride(0)`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字：auto 或 cann_ops_transformer。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B,S,topk+K-1]`（四维 query）或 `[T,topk+K-1]`（二维/三维 query） | `torch.int32` | query 所在 NPU 上的新输出。 |

所有 tensor 位于同一 NPU，采用与布局匹配的连续 ND 存储。
三个 metadata 参数统一使用 host 序列或连续 int64 设备张量，不混用；设备张量可在图回放时更新。
TND 空历史或全 padding 场景先调用 `prepare_pool_key_indexer_inputs`，
CANN 输出裁掉新增行后，再按原始 query 边界执行 packing。

## cann_ops_transformer 后端

公共适配层加载 `sgl_ops_transformer` 第三方包，调用 `torch.ops.cann_ops_transformer.pool_key_indexer`，并返回其中的 sparse indices。CANN 9.2 的 `ops_transformer` profile 会为 arch22 和 arch35 构建 vendor 算子及上游 Torch 绑定，不再需要单独安装 `csrc/attentions` 扩展。host 整数序列会转换为 CPU int64 tensor。二维/三维 query 直接转发，不转换成 BSND。

仅支持 BF16/FP16、`quant_mode=-1`、`return_value=False`。
无独立 `torch_native` 后端；`auto`/`cann_ops_transformer` 为同一实现。
测试使用可独立确定排名的数据，并覆盖空历史图回放回归。

## 调用示例

需先满足上述第三方包及 vendor 算子依赖；下例是最小 BSND 调用，不代表已完成 NPU 验收。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import pool_key_indexer

# BSND, one query/head, 16 complete pools and three tail tokens.
query = torch.zeros(1, 1, 1, 128, device="npu", dtype=torch.float16)
query[..., 0] = 1
keys = torch.zeros(1, 16, 1, 128, device=query.device, dtype=query.dtype)
keys[..., 0] = torch.arange(1, 17, device=query.device, dtype=query.dtype).view(1, 16, 1)
weights = torch.ones(1, 1, 1, device=query.device, dtype=query.dtype)
indices = pool_key_indexer(
    query, keys, weights, pool_tail_k=[3],
    topk=16, pool_size=4, layout_q="BSND", layout_k="BSND", impl="cann_ops_transformer",
)
assert indices.shape == (1, 1, 19) and indices.dtype == torch.int32
```
