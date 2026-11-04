# paged_mtp_attention_decode

## 硬件支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

```python
from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

def paged_mtp_attention_decode(
    *,
    query,
    k_cache,
    v_cache,
    seq_lens,
    cu_q_lens,
    block_tables,
    page_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    gamma,
    softmax_scale,
    max_kv_len,
    causal=False,
    impl="auto",
):
    ...
```

所有参数均仅限关键字传入。`auto` 选择 `triton`，不自动回退。

## 功能说明

面向 DSpark 固定宽度 draft block 的分页分组查询注意力。每个请求包含
G=`gamma` 个 query，批大小为 B，Hq=`num_q_heads`，Hkv=`num_kv_heads`，
D=`head_dim`，要求 Hq/Hkv 为整数。Q 使用 TND 布局，K/V 使用 FIA 的 ND 分页布局。

对于请求 b、query i、头 h 和 KV 位置 j，令 R=Hq/Hkv，P=page_size，
g=floor(h/R)，p=block_tables[b, floor(j/P)]，r=j mod P。
对应的 K/V 为 cache[p, r] 中从 g*D 开始的 D 个元素。

$$
S_{bhij}=\mathrm{softmax\_scale}\sum_d Q_{bG+i,h,d}K_{bj,g,d},\qquad
O_{bG+i,h,d}=\sum_{j\in A_{bi}}\mathrm{softmax}_{A_{bi}}(S_{bhi})_j V_{bj,g,d}.
$$

记 Lb=`seq_lens[b]`。非因果模式允许 `0 <= j < Lb`；因果模式还要求
`j <= Lb - G + i`，即 query 与包含当前 query block 的有效 KV 末端右对齐。
当可见集合为空时输出严格为零，包括 Lb=0 的填充请求。

`max_kv_len` 仅控制任务划分，有效读取范围始终由 `seq_lens` 决定。
未使用的缓存尾部不会参与计算。不支持变长 query block 和滑动窗口。
输入不会被修改。不同实现的浮点归约顺序和中间舍入可能存在差异。

## 参数与返回值

所有张量输入必须位于同一设备。Triton 要求 NPU 张量；参考实现接受 CPU 或 NPU
张量，但导入包和注册框架仍需要配置好的原生运行时、torch_npu、ACL 和可用 NPU。

| 参数 | 类型 | 形状 / 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `query` | Tensor | `[B*G, Hq, D]`，必填 | float16 或 bfloat16；TND 布局，支持非连续步长。有效数据必须有限。 |
| `k_cache` | Tensor | `[pages, P, Hkv*D]`，必填 | 与 query 类型一致；ND 布局，最后一维步长为 1，外层步长可非连续。有效 KV 数据必须有限。 |
| `v_cache` | Tensor | 与 k_cache 同形，必填 | 与 k_cache 相同的类型和布局约束。 |
| `seq_lens` | Tensor | `[B]`，必填 | 连续 int32；每个请求的有效 KV 末端，非负且不超过该行页表容量。0 表示填充请求。 |
| `cu_q_lens` | Tensor | `[B+1]`，必填 | 连续 int32；值必须等于 `arange(B+1)*G`。 |
| `block_tables` | Tensor | `[B, max_pages]`，必填 | int32 或 int64，支持非连续步长。有效 KV 范围内的条目必须指向实际缓存页；范围外条目可以无效，空请求整行均可无效。 |
| `page_size` | int | 必填 | 不小于 16 的 2 的幂，与缓存第二维一致。 |
| `num_q_heads` | int | 必填 | 正整数 Hq，可被 Hkv 整除。 |
| `num_kv_heads` | int | 必填 | 正整数 Hkv。 |
| `head_dim` | int | 必填 | D 在 `[16, 256]` 内，与 query 最后一维一致。 |
| `gamma` | int | 必填 | 正整数，每个请求固定的 query 宽度 G。 |
| `softmax_scale` | float | 必填 | 有限的分数缩放系数，通常为 `D**-0.5`。 |
| `max_kv_len` | int | 必填 | 非负调度界限，通常使用每个请求已分配的 KV 容量；不屏蔽或截断有效 KV。 |
| `causal` | bool | `False` | 启用右对齐因果掩码。 |
| `impl` | str | `"auto"` | `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | 形状 | 数据类型 / 设备 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | `[B*G, Hq, D]` | 与 query 一致 | 新分配的结果；B=0 返回空张量。输出步长不属于 API 契约。 |

公共 API 在分派前检查元数据。长度、偏移、页索引和有效数据有限性等张量值约束
由调用方保证；Triton 路径不会为验证这些约束将长度复制到 CPU。

## triton 后端

支持 NPU 上的 FP16/BF16 分页注意力和 NPU Graph 捕获。每次回放从设备缓冲区
重新读取长度、页表和 Q/K/V，无需在主机提取长度。

## torch_native 后端

独立精度参考实现，使用 FP32 中间值，输出转换为 query.dtype。
该实现会在主机读取长度和偏移，不支持 NPU Graph 捕获。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

device = "npu:0"
args = dict(
    query=torch.randn(7, 16, 64, dtype=torch.bfloat16, device=device),
    k_cache=torch.randn(2, 128, 256, dtype=torch.bfloat16, device=device),
    v_cache=torch.randn(2, 128, 256, dtype=torch.bfloat16, device=device),
    seq_lens=torch.tensor([129], dtype=torch.int32, device=device),
    cu_q_lens=torch.tensor([0, 7], dtype=torch.int32, device=device),
    block_tables=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    page_size=128,
    num_q_heads=16,
    num_kv_heads=4,
    head_dim=64,
    gamma=7,
    softmax_scale=64**-0.5,
    max_kv_len=256,
    causal=True,
)
output = paged_mtp_attention_decode(**args)
reference = paged_mtp_attention_decode(**args, impl="torch_native")
torch.testing.assert_close(output, reference, atol=0.004, rtol=0.03)
```
