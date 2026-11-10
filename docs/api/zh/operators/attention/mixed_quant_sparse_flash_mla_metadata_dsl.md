# mixed_quant_sparse_flash_mla_metadata_dsl

## 芯片支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | 不支持 | 支持 |

## 函数原型

`from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata_dsl`

```python
def mixed_quant_sparse_flash_mla_metadata_dsl(
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

## 功能说明

MQSMLA 的 AICPU 调度算子：根据两侧 TopK 长度生成整行或 FlashDecode 分片执行计划。

`ori_topk_length` 与 `cmp_topk_length` 是每个 Query 行两侧的有效稀疏长度；`has_ori_kv` 与 `has_cmp_kv` 应与主算子提供的 KV 侧一致。`num_heads_q=64`、`num_heads_kv=1`、`head_dim=512`、`quant_mode=1`。

## 参数

T 为查询行数，B 为请求数，P 为物理页数，S 为每页槽位数。

| 参数 | 默认值 | 类型、形状与含义 |
| --- | --- | --- |
| `ori_topk_length` | 必填 | 必需的连续 INT32 [T,1]，每个查询原始稀疏索引的有效前缀长度。 |
| `cmp_topk_length` | 必填 | 必需的连续 INT32 [T,1]；has_cmp_kv=False 时传入全零张量。 |
| `cu_seqlens_q` | `None` | 可选 INT32 [B+1]，各请求查询行的累计边界。 |
| `seqused_q` | `None` | 可选 INT32 [B]，兼容参数；调度使用 cu_seqlens_q。 |
| `seqused_ori_kv` | `None` | 可选 INT32 [B]，原始 KV 长度；保留该兼容参数。 |
| `seqused_cmp_kv` | `None` | 可选 INT32 [B]，压缩 KV 长度；保留该兼容参数。 |
| `batch_size` | `None` | 可选 int，兼容参数；B 由 cu_seqlens_q 推导，省略边界时按单请求处理。 |
| `max_seqlen_q` | `None` | 可选 int，兼容参数；查询行数由长度张量的形状确定。 |
| `max_seqlen_ori_kv` | `None` | 可选 int，原始 KV 容量（兼容参数）。 |
| `max_seqlen_cmp_kv` | `None` | 可选 int，压缩 KV 容量（兼容参数）。 |
| `num_heads_q` | 必填 | int，查询头数，使用 64。 |
| `num_heads_kv` | 必填 | int，共享 MLA KV 头数，使用 1。 |
| `head_dim` | 必填 | int，逻辑头维度，使用 512。 |
| `quant_mode` | 必填 | int，当前支持的量化配置使用 1。 |
| `layout_q` | `'TND'` | str，使用 TND（展平查询 token）。 |
| `layout_kv` | `'PA_BBND'` | str，使用 PA_BBND 打包分页 KV。 |
| `has_ori_kv` | `True` | bool，当前实现必须为 True。 |
| `has_cmp_kv` | `True` | bool，消费该 metadata 的 attention 是否包含压缩 KV。 |
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
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata_dsl

cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
length = torch.tensor([[128]], device="npu", dtype=torch.int32)
metadata = mixed_quant_sparse_flash_mla_metadata_dsl(
    length,
    torch.zeros_like(length),
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
python -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata_dsl.py --npu-device=0 --enable-perf=false -v -s
```
