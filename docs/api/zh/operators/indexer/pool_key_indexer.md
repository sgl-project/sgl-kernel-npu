# pool_key_indexer

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclnn` | 未验证 | 已验证 |

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

`auto` 固定选择 `aclnn`，首次执行时延迟加载 `attentions` 扩展；不进行后端回退。

## 功能说明

对 pooled key 打分并选择排名靠前的 pool，将其展开为 token 位置，再追加末尾尚未组成完整 pool 的 token。

输出为 int32 索引，不返回分数；未使用的 tail 位置填为 `-1`。打分、掩码和选中 pool 的排列由 vendor 算子决定，不保证 pool 顺序稳定。输入不被修改。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype / 位置 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `query` | `torch.Tensor` | `[B,S,H,D]` / `[T,H,D]` / `[T,H*D]` | `float16` / `bfloat16` | 必填 | BSND / TND query；二维输入直接转发，不在 binding 中 reshape。 |
| `pool_key` | `torch.Tensor` | `[B,P,1,D]` (BSND); layout-dependent for PA | 与 query 相同 | 必填 | pool key；PA 分页布局须配套页表。 |
| `weights` | `torch.Tensor` | `[B,S,H]` (BSND) / `[T,H]` (TND) | 与 query 相同 | 必填 | 逐 query、逐 head 的 indexer 权重。 |
| `pool_tail_k` | `Sequence[int]` | `[B]` | host 整数 | 必填 | 每请求未成 pool 的 tail token 数，范围 `[0,pool_size)`；传 host list/tuple，不得传 NPU tensor。 |
| `actual_seq_q` | `Sequence[int]` or `None` | `[B]` or — | host 整数 | `None` | packed TND 的 query 累积结束位置；None 转为空 aclIntArray。 |
| `actual_seq_k` | `Sequence[int]` or `None` | `[B]` or — | host 整数 | `None` | 分页场景逐请求的有效 pooled-key 长度；None 转为空 aclIntArray。 |
| `block_table` | `torch.Tensor` or `None` | `[B,M]` or — | `int32` | `None` | 分页 key 的 NPU 页 ID；None 转为空 optional tensor。 |
| `topk` | `int` | — | — | `2048` | token 级选取宽度，至少为 pool_size；按 pool_size 的倍数使用，并提供足够有效 pool。 |
| `pool_size` | `int` | — | — | `16` | 每个完整 pool 对应的 token 数，须为正数。 |
| `layout_q` | `str` | — | — | `"BSND"` | query 布局；示例涉及 BSND，测试另含 TND。 |
| `layout_k` | `str` | — | — | `"BSND"` | key 布局；测试配置使用 BSND 或 PA_BBND。 |
| `mask_mode` | `int` | — | — | `3` | 透传 vendor mask mode；当前接入使用模式 3。 |
| `quant_mode` | `int` | — | — | `-1` | 仅支持 -1；不对外暴露 descale tensor。 |
| `return_value` | `bool` | — | — | `False` | 必须为 False：schema 仅返回索引。 |
| `key_stride0` | `int` | — | — | `-1` | 透传 vendor key stride；-1 表示未指定，Python 层不推导。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字：auto 或 aclnn。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B,S,topk+K-1]`（四维 query）或 `[T,topk+K-1]`（二维/三维 query） | `torch.int32` | query 所在 NPU 上的新输出。 |

所有 tensor 须位于同一 NPU，采用与布局匹配的连续 ND 存储；序列长度和 `pool_tail_k` 保持 host 整数序列。布局前缀检查允许 `TND*`、`PA_*`、`BSND*`，但不代表所有后缀、shape 或 stride 均被 vendor 支持。

## aclnn 后端

公共适配层调用 `torch.ops.attentions.pool_key_indexer`，由 C++ binding 转发到 `aclnnPoolKeyIndexer`。需要单独构建并安装 `csrc/attentions` 扩展，以及提供该 vendor 算子的 CANN/custom-op 环境；仅安装主 kernel 包不足以运行。二维/三维 query 直接转发，不转换成 BSND。

仅支持 BF16/FP16、`quant_mode=-1`、`return_value=False`。无独立 `torch_native` 后端；`auto`/`aclnn` 是同一实现，不能相互作为精度基准。现有测试使用独立的确定性排名数据，尚无该 binding 的硬件精度或性能验收结论。

## 调用示例

需先满足上述扩展及 vendor 算子依赖；下例是最小 BSND 调用，不代表已完成 NPU 验收。

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
    topk=16, pool_size=4, layout_q="BSND", layout_k="BSND", impl="aclnn",
)
assert indices.shape == (1, 1, 19) and indices.dtype == torch.int32
```
