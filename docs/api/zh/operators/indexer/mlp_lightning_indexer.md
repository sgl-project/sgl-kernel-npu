# mlp_lightning_indexer

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `auto` / `ascendc` | 支持实现 | 不支持 |
| `torch_native` | 独立 PyTorch 参考实现 | 不支持 |

此表表示实现范围，不表示本次新增的硬件验证结果。在 arch22 Linux/CANN 环境执行
根目录 `python3 setup.py --arch-version 2201 --ops mlp_lightning_indexer`，然后安装
生成的 wheel。arch22 的 `--ops all` 也包含此算子。
安装步骤见[构建指南](../../../../../python/sgl_kernel_npu/README.md)。

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer

def mlp_lightning_indexer(
    query, key, weights,
    cur_seq_lengths_query=None, cur_seq_lengths_key=None,
    block_table=None, init_tensor=None, local_tensor=None,
    layout_query="BSND", layout_key="PA_BSND", sparse_count=2048,
    kv_block_len=1, q_block_len=1, init_num=0, local_num=0,
    sparse_mode=3, pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807, return_value=False, *, impl="auto",
):
    ...
```

`auto` 选择 `ascendc`，调用 `torch.ops.npu.mlp_lightning_indexer`，不自动回退。
保留原生名称、默认值和位置参数顺序。配套 LongCat2 调用方传入 int64 长度，
此包装层不转换长度 dtype。

## 功能说明

为 LongCat2 稀疏注意力选择 key block 索引。对 query 位置 `q`、key 位置 `t`、
query head `h` 及 key block 宽度 `L = kv_block_len`：

```text
token_score(q,t) = sum_h weights[q,h] * max(dot(query[q,h,:], key[t,0,:]), 0)
block_score(q,b) = sum_{t in block b} token_score(q,t) / L
```

分页 key 通过 `block_table` 还原。返回索引为请求内的逻辑 block 编号，不是物理页号；
`L=1` 时等同 token 索引。稀疏模式 0 使用完整 key 序列作为候选；模式 3 为右对齐 causal：
逻辑 query 位置 `q` 的可见 token 数为 `key_length - query_length + q + 1`，
候选 block 编号小于 `ceil(visible_count / L)`。

开头的 `init_num / L` 个 block，以及起点位于 `visible_count - local_num` 或之后的
block 获得优先分数；local 计数为零时不启用 local 优先级。
提供 `init_tensor`、`local_tensor` 时，分别以逐请求计数覆盖对应
标量。普通分数的上限为 `float32_max / 4`，优先分数为 `float32_max / 2`，
从候选中选择最大的 `sparse_count` 项。不增加额外的对角线优先规则；
同分项的顺序不是额外的 API 保证。

`return_value=True` 时，选中分数截断到最大 65504 后转为 query dtype，bfloat16 也沿用
该上限。缺失候选的 index 为 `-1`、value 为 `-inf`。`return_value=False` 时，
仍返回具有相同完整 shape 的两个 Tensor；**values 的内容无定义，不应消费或数值比较**。

TND 模式的两个长度数组均为 int64，长度为 `B+1`，包含前导零。
逻辑 query 长度为零的请求仍占一个 query、weights 和 output 的物理槽，
因此物理 query 行数为 `T = sum_r max(query_length[r], 1)`；
逻辑累计边界仍使用原始 query 长度。零 query 槽的 indices 为 padding。
这是现有 kernel 的约定，不是新增的通用 ragged layout。

## 参数与返回值

Tensor 输入沿用同一 NPU、连续存储的调用约定。Q/K dtype 相同，`N_k=1`、head 维度
`D=128`。以下描述记录现有契约，公共分发层不增加新的输入校验或类型转换。

| 参数 | 类型 / Shape / Dtype | 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | Tensor，FP16/BF16；TND `[T,N_q,128]`，BSND `[B,S_q,N_q,128]` | 必填 | Query，TND 包含零 query 的物理槽。 |
| `key` | Tensor，与 query 同 dtype；PA_BSND `[P,page_size,1,128]`，TND `[T_k,1,128]`，BSND `[B,S_k,1,128]` | 必填 | Key；原生页大小为 `(0,1024]` 内的 16 倍数。 |
| `weights` | Tensor，float32；TND `[T,N_q]`，BSND `[B,S_q,N_q]` | 必填 | 每个 query、每个 head 的分数权重。 |
| `cur_seq_lengths_query` | 可选 Tensor，int64；TND `[B+1]` | `None` | TND 必填，逻辑 query 累计边界；不执行 int32 转换。 |
| `cur_seq_lengths_key` | 可选 Tensor，int64；TND query 路径 `[B+1]` | `None` | TND 路径的逻辑 key 累计边界；TND 或分页 key 需要提供。 |
| `block_table` | 可选 Tensor `[B,max_pages]`，int32 | `None` | 逻辑页到物理页的映射；PA_BSND 必填，非分页 key 不传。 |
| `init_tensor` | 可选 Tensor `[B]`，int32 | `None` | 每请求的开头 token 数，覆盖 `init_num`。 |
| `local_tensor` | 可选 Tensor `[B]`，int32 | `None` | 每请求的 local token 数，覆盖 `local_num`。 |
| `layout_query` | str | `"BSND"` | 历史原生布局：BSND 或 TND；参考实现范围见后文。 |
| `layout_key` | str | `"PA_BSND"` | PA_BSND，或与 query 相同的非分页布局。 |
| `sparse_count` | int | `2048` | 输出宽度；原生支持 `1..2048`，或不超过 8192 的 128 倍数。 |
| `kv_block_len` | int | `1` | 打分 block 宽度：1、2、4、8、16；与物理页大小不同。 |
| `q_block_len` | int | `1` | 现有原生实现要求为 1。 |
| `init_num` | int | `0` | 开头 token 数：`0..16`，为 `kv_block_len` 的倍数。 |
| `local_num` | int | `0` | Local token 数：`0..2048`，为 `kv_block_len` 的倍数。 |
| `sparse_mode` | int | `3` | 0：完整候选序列；3：右对齐 causal 候选。 |
| `pre_tokens` | int | `9223372036854775807` | 保留的原生属性，当前 kernel 不使用它计算窗口 mask。 |
| `next_tokens` | int | `9223372036854775807` | 保留的原生属性，当前 kernel 不使用它计算窗口 mask。 |
| `return_value` | bool | `False` | 分数输出是否具有定义的内容。 |
| `impl` | str | `"auto"` | 仅限关键字：`auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 / Shape / Dtype | 说明 |
| --- | --- | --- |
| `indices` | Tensor，int32；TND `[T,1,sparse_count]`，BSND `[B,S_q,1,sparse_count]` | 选中的逻辑 key block 编号，padding 为 `-1`。 |
| `values` | Tensor，query dtype，与 indices 同 shape | 仅 `return_value=True` 时是选中分数；否则内容无定义。 |

输出为 query 设备上的新 Tensor，不修改输入。

## ascendc 后端

沿用官方 CANN OpDef、shape/dtype 推导、tiling 和模板 kernel，通过本仓库的 Torch
adapter 调用 `aclnnSglMlpLightningIndexerGetWorkspaceSize` 和 `aclnnSglMlpLightningIndexer`。
CANN 管理 tiling 与 kernel 调度，不再使用旧直接启动路径的私有 tiling cache。
公共 API 保留历史 BSND 签名和分支，本次迁移不扩展该分支。TND + PA_BSND 是配套
LongCat2 路径；继续由调用方提供 int64 长度。

ACLNN 迁移后的 NPU 精度、性能和已有 graph capture/replay 用例需要在对应构建上重新
验证。此前直接启动路径的 graph 结果不能作为新链路的验证结果。

## torch_native 后端

使用 PyTorch 独立计算分数、block 归约、候选优先级和选择。当前精度测试覆盖 TND query
搭配 PA_BSND 或 TND key，包括现有零 query 物理槽约定。参考实现也保留 BSND 分支，
两个长度数组均显式使用 `B+1` 累计边界。原生 BSND host 要求长度数组为 `B` 项，
而 device 分支读取累计边界；这一继承的差异不属于本次 TND 验收范围。
参考实现不会在长度为 `None` 时补造默认长度。

它用于精度参考，不是原生后端的自动回退。与原生结果一样，
`return_value=False` 时不比较 values。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer

device = "npu:0"
torch.npu.set_device(device)
query = torch.randn(1, 8, 128, device=device, dtype=torch.float16)
key = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float16)
weights = torch.ones(1, 8, device=device, dtype=torch.float32)
q_lengths = torch.tensor([0, 1], device=device, dtype=torch.int64)
k_lengths = torch.tensor([0, 16], device=device, dtype=torch.int64)
pages = torch.tensor([[0]], device=device, dtype=torch.int32)
indices, values = mlp_lightning_indexer(
    query, key, weights,
    cur_seq_lengths_query=q_lengths, cur_seq_lengths_key=k_lengths,
    block_table=pages, layout_query="TND", layout_key="PA_BSND",
    sparse_count=8, return_value=True, impl="ascendc",
)
# 两个输出 shape 均为 [1, 1, 8]；使用 impl="torch_native" 选择参考实现。
```
