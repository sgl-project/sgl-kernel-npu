# flash_decode_bnsd_with_gqa_share_sparse

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import flash_decode_bnsd_with_gqa_share_sparse

def flash_decode_bnsd_with_gqa_share_sparse(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
    *,
    impl="auto",
):
    ...
```

当 `use_native=True`、`sink is None`、`D=128`、`block_size=128` 且当前构建中原生算子可用时，
`auto` 选择 `ascendc`；否则选择 `triton`。显式指定 `triton`、`ascendc` 或 `torch_native`
会直接选择对应后端。后端执行出错会向外抛出，不会换后端重试。

## 功能说明

在调用方预先选好的 KV block 上计算稀疏注意力；本算子不负责 top-k 选块。每个 GQA 组内的
query 头共享一个 KV 头及其选块列表。`B` 是 query 行数：decode 时为 batch，verify/prefill 时
为展开后的 query 数。`QH`、`KVH` 分别为 query 和 KV 头数，`G=QH/KVH`，`D` 为 head
维度，`BS=block_size`，`NB` 为物理 cache 页数。query 头 `h` 使用 KV 头 `kvh=h//G`。

对第 `b` 行，选中的逻辑块 `lb=topk_idx[kvh,b,t]` 提供块内偏移 `n=0..BS-1` 的 token，
仅 `lb>=0` 且 `lb*BS+n < seq_lens[b]` 的位置参与计算。负块号表示空槽，末尾不满一块的
无效 token 会被掩码。输入索引为 `[QH,B,T]` 时，只取每个 GQA 组的第一个 query 头的选块
列表。重复块会重复参与计算，不做去重。

物理页号 `phys(t)` 由以下两种页映射之一给出：

- **块页表：**`block_table[b,lb]`。
- **直接映射：**`req_to_token[req_pool_indices[b],lb*BS] // BS`。只读取块首 token slot，
  每个逻辑块必须映射到一个完整物理 cache 页。`sanitize_page_ids=True` 时，将所得物理页号
  夹到 `[0,num_pages-1]`。

设 $\mathcal{S}_{b,kvh}$ 为上述有效选中 token 的集合（保留重复项），每个 query 头的计算为：

$$
s_{tn} = \mathrm{sm\_scale}\; q_{b,h}^{T} K_{\mathrm{phys}(t),n,kvh}, \qquad
z_{\mathrm{sink}} =
\begin{cases}
\exp(\mathrm{sm\_scale}\; q_{b,h}^{T} sink_h), & sink \ne \mathrm{None}, \\
0, & sink = \mathrm{None},
\end{cases}
$$

$$
out_{b,h} =
\frac{\sum_{(t,n)\in\mathcal{S}_{b,kvh}} \exp(s_{tn}) V_{\mathrm{phys}(t),n,kvh}}
{z_{\mathrm{sink}} + \sum_{(t,n)\in\mathcal{S}_{b,kvh}} \exp(s_{tn})}
$$

求和仅包含有效 token。sink 提供一个与 query 相关的 softmax logit，其 value 为零；输入是
`[QH,D]` 向量，而非预先计算好的标量 logit。没有有效 token 时输出为零，包括提供 sink 的
情况。输出的 shape 和 dtype 与 `q` 相同。

本算子没有额外的因果掩码或 query 位置参数。实现因果 verify/prefill 时，调用方需将 query
展开成行，并通过 `seq_lens` 提供每行可见的 KV 前缀长度。同一请求的多个 query 可以共享
`req_pool_indices` 中的请求行号。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[B, QH, D]` | `torch.float16`、`torch.bfloat16` | 必填 | Query；`B`、`QH`、`D` 为正，且 `QH % KVH == 0`。 |
| `sink` | `torch.Tensor` / `None` | `[QH, D]` | 同 `q` | 必填（可传 `None`） | 可选 sink 向量；shape、dtype 须符合表中要求。`ascendc` 仅接受 `None`。 |
| `k_cache_bnsd` | `torch.Tensor` | `[NB, BS, KVH, D]` | 同 `q` | 必填 | paged key cache；`NB` 为物理页数。 |
| `v_cache_bnsd` | `torch.Tensor` | 同 `k_cache_bnsd` | 同 `q` | 必填 | paged value cache，shape 与 dtype 必须与 K 相同。 |
| `block_table` | `torch.Tensor` / `None` | `[B, max_num_blocks]` | `int32`、`int64` | 必填（与 direct 二选一） | 逻辑块号到物理页号的映射；使用直接映射时传 `None`。 |
| `seq_lens` | `torch.Tensor` | `[B]` | `int32`、`int64` | 必填 | 每行可见的 KV token 数量，不是最后一个 token 的下标。要求连续、非负，不超过页映射容量及有符号 int32 范围。 |
| `block_size` | `int` | — | — | 必填 | 每个 cache 页的 token 数，须与 cache 第二维一致。Triton 要求为正的 2 的幂；`ascendc` 要求为 128。 |
| `topk_idx` | `torch.Tensor` | `[KVH 或 QH, B, T]` | `int32`、`int64` | 必填 | 选中的逻辑块号，`T > 0`；用 `-1` 表示空槽。非负块号必须在页映射容量内。首维为 `QH` 时取每个 GQA 组的第一个头。 |
| `sm_scale` | `float` / `None` | — | — | `None` | softmax 缩放；`None` 表示 `D ** -0.5`。 |
| `num_topk_chunks` | `int` / `None` | — | — | `None` | 仅 Triton：正的 2 的幂，且不超过 `T`。`None` 根据 batch、KV 头数、`T` 和 vector core 数选择；小 batch、短选块列表使用单 chunk。 |
| `max_num_topk_chunks` | `int` | — | — | `8` | 仅 Triton：自动分块数的正整数上限；显式指定 `num_topk_chunks` 时忽略。 |
| `req_to_token` | `torch.Tensor` / `None` | `[num_requests, max_context]` | `int32`、`int64` | `None` | 请求行和 token 位置到物理 token slot 的直接映射；与 `block_table` 互斥，需同时提供 `req_pool_indices`。 |
| `req_pool_indices` | `torch.Tensor` / `None` | `[B]` | `int32`、`int64` | `None` | 每个 query 行在 `req_to_token` 中的请求行号，要求连续；与 `req_to_token` 同时提供或同时为 `None`。 |
| `max_num_blocks` | `int` / `None` | — | — | `None` | 直接映射时必填且为正，`max_num_blocks * BS <= req_to_token.shape[1]`；块页表路径忽略。 |
| `num_pages` | `int` / `None` | — | — | `None` | 直接映射时必填且为正，设为 `NB`；清洗物理页号时用作上界。块页表路径忽略。 |
| `sanitize_page_ids` | `bool` | — | — | `False` | 是否把 direct 路径的物理页号夹到 `[0, num_pages-1]`；页表路径忽略。 |
| `topk_blocks_per_step` | `int` | — | — | `1` | 仅 Triton：每步融合的选中块数。有效值为 `max(1, int(value))`，须为 2 的幂。`D=128, BS=128` 时保持为 1；见下方后端限制。 |
| `prefetch_idx` | `bool` | — | — | `False` | 仅 Triton：预取块号与物理页号；仅在 `topk_blocks_per_step` 有效值为 1 时生效。 |
| `num_warps` | `int` / `None` | — | — | `None` | Triton launch 参数；`None` 使用 4。 |
| `num_stages` | `int` / `None` | — | — | `None` | Triton launch 参数；`None` 使用 2。 |
| `use_native` | `bool` | — | — | `True` | 仅 `auto` 使用：允许在上述条件满足时选择原生后端。设为 `False` 时 `auto` 选择 Triton；显式后端忽略。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"ascendc"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, QH, D]` | 同 `q` | 新建输出，与 `q` 同设备；输入全部只读。 |

**调用方约定：**所有 Tensor 须在同一 NPU 上。跨后端调用时使用连续输入，尤其是 Triton 未传入
stride 的 `seq_lens`、`req_pool_indices`。`KVH` 和页映射各维须为正，各 Tensor 的 shape 须
符合表中对应关系。非负逻辑块号必须小于 `block_table.shape[1]` 或 `max_num_blocks`，即使
该块位于可见前缀之外。请求行号和物理页号必须有效。页号清洗仅夹取直接映射所得的物理页号，
不校验逻辑块号、请求行号或块页表内容。query、sink 及有效 K/V 数据须为有限值。

**被断言的约束：**Triton 和 AscendC 检查 query/cache dtype 与 K/V shape 一致，并在
GQA 索引归一化时检查 topk 首维。Triton 另检查 cache 维度、batch 大小、GQA 整除关系、页映射参数及分块/融合
参数。AscendC 另要求 `sink=None, D=128, BS=128`。PyTorch 参考实现未重复这些断言；
无论选择哪个后端，调用方都须遵守表格和共用约定。

## triton 后端

在选中块上计算在线 softmax。`num_topk_chunks` 将选块列表分片并行计算；单 chunk 省去合并
launch，多 chunk 则需临时部分输出。中间量以 fp32 累加，softmax 权重在乘 V 前转换为 K/V
的 dtype，部分输出以 `q.dtype` 保存，因此分块数可能影响浮点结果。

`D=128, BS=128` 时，`topk_blocks_per_step=2` 超出已测试 A2/A3 的 UB 容量，编译会报
`ub overflow`；Triton 下请保持为 1。`auto` 不会在编译失败后更换 launch 参数重试。

## ascendc 后端

调用 `csrc/sparse_attention_score` 提供的 `torch.ops.npu.npu_sparse_attention_score`。
要求 arch22 构建包含原生算子，且 `D=128`、`BS=128`、`sink=None`。显式请求时，原生算子
不可用会报错，不支持的尺寸或非空 sink 会被断言拒绝。该适配层的索引预处理也依赖 Triton。

接受两种页映射，并将每 query 头的索引转换为 GQA 共享索引。固定使用 `inner_precise=4`
进行高精度累加。接受但忽略分块、块融合、预取和 launch 参数。

## torch_native 后端

独立的 PyTorch 精度参考，score、权重和累加均使用 fp32，最后转换为 `q.dtype`。仅收集选中块，
按 query 行和 KV 头循环，适合正确性对比。忽略分块、块融合、预取、launch 参数和 `use_native`。

不同后端的归约顺序、权重舍入和中间存储不同，结果不保证逐位一致。相消和较大的 V 数值可能
放大差异，应按实际 dtype 和负载选择精度容差。Triton 将 `seq_lens` 转为 int32，参考实现
按 int64 处理；合法输入需处于有符号 int32 范围内。

## 调用示例

需要安装本包及 NPU 运行依赖。示例使用块页表；改用直接映射时，传 `block_table=None` 并提供
`req_to_token`、`req_pool_indices`、`max_num_blocks` 和 `num_pages`。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_decode_bnsd_with_gqa_share_sparse

B, QH, KVH, D, BS, NB = 1, 8, 1, 128, 128, 8

torch.manual_seed(0)
q = torch.randn(B, QH, D, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(NB, BS, KVH, D, dtype=torch.bfloat16, device="npu")
v_cache = torch.randn_like(k_cache)
topk_idx = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32, device="npu")  # [KVH, B, topk]
seq_lens = torch.tensor([385], dtype=torch.int32, device="npu")

# 逻辑块号与物理页号相同。
block_table = torch.arange(NB, dtype=torch.int32, device="npu").view(B, NB)

out_auto = flash_decode_bnsd_with_gqa_share_sparse(
    q, None, k_cache, v_cache, block_table, seq_lens, BS, topk_idx,
    sm_scale=D**-0.5,
    num_topk_chunks=1,
)
out_ref = flash_decode_bnsd_with_gqa_share_sparse(
    q, None, k_cache, v_cache, block_table, seq_lens, BS, topk_idx,
    sm_scale=D**-0.5,
    num_topk_chunks=1,
    impl="torch_native",
)
print(out_auto.shape, out_auto.dtype)  # torch.Size([1, 8, 128]) torch.bfloat16
torch.testing.assert_close(out_auto, out_ref, atol=5e-2, rtol=5e-2)
```
