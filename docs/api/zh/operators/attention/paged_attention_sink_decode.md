# paged_attention_sink_decode

## 硬件支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | 未验证 | 支持 |
| `torch_native` | 支持 | 支持 |

Triton 实现在 arch22 上未验证：其向量路径按 A5 编译选项调优。在 arch22 上
使用前请先验证。

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention import paged_attention_sink_decode


def paged_attention_sink_decode(
    *,
    q,
    k_cache,
    v_cache,
    seqlens,
    block_tables,
    sinks=None,
    gqa_interleave=False,
    softmax_scale=None,
    impl="auto",
): ...
```

所有参数均为关键字参数。`auto` 直接选择 `triton`，不做回退。

## 功能描述

带每个查询头一个可学习 sink logit 的分页分组查询 decode 注意力。每个请求只有
一行查询，其位置就是最后一个 KV 条目，因此整行 KV 均可见，无需因果掩码。
sink 以零值向量参与 softmax 分母，只会降低 KV 的注意力质量。

对请求 b、头 h、KV 位置 j < Lb=`seqlens[b]`，记 P=`page_size`、g 为 h 对应的
KV 头、`p = block_tables[b, floor(j / P)]`、`r = j mod P`：

$$
S_{bj} = \mathrm{softmax\_scale}\sum_d q_{bhd}\,k_{p,r,g,d},\qquad
O_{bhd} = \frac{\sum_{j} e^{S_{bj}}\,v_{p,r,g,d}}
{\sum_{j} e^{S_{bj}} + e^{s_h}} .
$$

`gqa_interleave` 为真时 `g = h mod Hkv`，否则 `g = h // (Hq / Hkv)`。
`Lb=0` 的请求输出零行。

计算采用 FP32 在线 softmax 累加，输出转换为 query dtype。不同实现的归约顺序与
中间舍入可能不同，KV 切分路径还会改变跨切分的归约顺序。

## 参数与返回值

所有张量须位于同一设备。Triton 实现要求 NPU 存储；参考实现也接受 CPU 张量，
但导入包仍需已配置的 native runtime、torch_npu、ACL 以及可用的 NPU。

| 参数 | 类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `q` | Tensor | 必填 | 连续的 FP16/BF16 decode 查询 `[B, Hq, D]`。 |
| `k_cache` | Tensor | 必填 | 连续的 key 缓存 `[pages, P, Hkv, D]`，dtype 与 q 一致。 |
| `v_cache` | Tensor | 必填 | 与 `k_cache` 同形状、同 dtype。 |
| `seqlens` | Tensor | 必填 | 连续 int32 `[B]`，每个请求的 KV 长度（含当前 token），各请求可不同。 |
| `block_tables` | Tensor | 必填 | 连续 int32/int64 物理页 ID `[B, max_pages]`；第 b 行须覆盖 `ceil(seqlens[b] / P)` 个页。 |
| `sinks` | Tensor | `None` | FP16/BF16 `[Hq]` sink logits，dtype 与 query 一致；`None` 关闭 sink。 |
| `gqa_interleave` | bool | `False` | 使用交错的查询头到 KV 头映射。 |
| `softmax_scale` | float | `None` | 有限正数 QK 缩放系数；`None` 取 `D**-0.5`。 |
| `max_kv_len` | int | `None` | 调用方已知的最长 KV 行。提供后切分判定与页表校验直接使用该值，无需设备到 host 的读取；须为正整数。 |
| `impl` | str | `"auto"` | `"auto"`、`"triton"` 或 `"torch_native"`。 |

其他约束：`Hq` 为正且可被 `Hkv` 整除；`P` 为不小于 16 的 2 的幂；`D` 为
`[16, 256]` 内的整数。

| 返回值 | 类型 | 形状 | dtype / 设备 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | `[B, Hq, D]` | 与 q 一致 | 新分配的结果；`B=0` 返回空张量。 |

## triton 后端

- 默认路径把每个（请求、KV 头）任务分配给 cube 核；同一组的 `Hq/Hkv` 个查询头
  共享每块 K/V，sink 参与在线最大值与分母的初始化。每个任务流式扫描整行 KV。
- 当 grid 无法填满 cube 核且 KV 较长时切换到两阶段 flash decode：触发条件为
  `B * Hkv < 0.4 * cube 核数` 且最长 KV 行不小于 256（GQA 需不小于 2048），
  切分数为 `max(1, min(cube 核数 // (B * Hkv), max_kv_len // 256))`。
- 第一阶段在 cube 核上运行，按（请求、KV 头、切分）写出局部归一化的部分输出及
  其 log-sum-exp；此阶段不加入 sink，避免每个切分重复计数。第二阶段在 vector
  核上运行，按 log-sum-exp 重新加权各切分，并把 sink 作为额外的零值分母项只加
  一次。
- 切分按页对齐，避免 KV 块跨越两个逻辑页。仅当 batch 属于切分候选
  （`B * Hkv` 小于 0.4 倍 cube 核数）且未提供 `max_kv_len` 时，才把最长 KV
  长度拷贝到 host，用于选择切分数并校验页表宽度；提供 `max_kv_len` 可完全
  省去该读取；其余 batch 的元数据全部留在设备上，kernel 会把页索引
  钳制到页表宽度，因此调用方须按实际长度分配 `block_tables`。切分路径每次调用
  都会分配 FP32 工作区，不支持 NPU Graph 捕获。
- arch35 上的 Triton-Ascend 3.6 支持非 2 的幂 head_dim 与 GQA 分组；arch22
  仍未验证。

## torch_native 后端

- 按 block table 收集每个请求的有效页，按所选映射展开共享 KV 头，并逐行计算
  FP32 softmax 注意力。
- 将 sink logit 作为额外的分数拼接后丢弃其值列，使其只扩大分母；`Lb=0` 的请求
  输出零行。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention_sink_decode

device = "npu:0"
q = torch.randn(2, 16, 64, dtype=torch.bfloat16, device=device)
k_cache = torch.randn(6, 128, 4, 64, dtype=torch.bfloat16, device=device)
v_cache = torch.randn_like(k_cache)
seqlens = torch.tensor([129, 300], dtype=torch.int32, device=device)
block_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
sinks = torch.randn(16, dtype=torch.bfloat16, device=device)

output = paged_attention_sink_decode(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    seqlens=seqlens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
)
reference = paged_attention_sink_decode(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    seqlens=seqlens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
    impl="torch_native",
)
torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
```
