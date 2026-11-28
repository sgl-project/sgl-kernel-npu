# paged_attention_sink_prefill

## 硬件支持

| 实现 | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | 未验证 | 支持 |
| `torch_native` | 支持 | 支持 |

Triton 实现在 arch22 上未验证：kernel 传递的是 A5 的 CV 流水线编译选项，
页聚合路径还依赖 Triton-Ascend 的 CANN 扩展。在 arch22 上使用前请先验证。

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill


def paged_attention_sink_prefill(
    *,
    q,
    k_cache,
    v_cache,
    cu_q_lens,
    block_tables,
    seqlens_kv=None,
    sinks=None,
    gqa_interleave=False,
    softmax_scale=None,
    impl="auto",
): ...
```

所有参数均为关键字参数。`auto` 直接选择 `triton`，不做回退。

## 功能描述

对打包（packed）请求计算带因果掩码的分页分组查询 prefill 注意力，每个查询头
带一个可学习的 sink logit。sink 以零值向量参与 softmax 分母，因此只会降低
KV 的注意力质量。

对请求 b、查询行 i、头 h、KV 位置 j，记 P=`page_size`、g 为 h 对应的 KV 头、
Lb 为 `seqlens_kv[b]`（省略 `seqlens_kv` 时取该请求的查询长度）、
Cb=Lb-Lq_b 为缓存前缀长度。新查询位于缓存前缀之后，因此可读取的 KV 位置为
`0 <= j <= Cb + i`。令 `p = block_tables[b, floor(j / P)]`、`r = j mod P`：

$$
S_{bj} = \mathrm{softmax\_scale}\sum_d q_{bihd}\,k_{p,r,g,d},\qquad
O_{bihd} = \frac{\sum_{j} e^{S_{bj}}\,v_{p,r,g,d}}
{\sum_{j} e^{S_{bj}} + e^{s_h}} .
$$

`gqa_interleave` 为真时 `g = h mod Hkv`，否则 `g = h // (Hq / Hkv)`。每个查询
至少能读到自身位置，因此不会出现可见 KV 为空的查询。

计算采用 FP32 在线 softmax 累加，输出转换为 query dtype。不同实现的归约顺序与
中间舍入可能不同。

## 参数与返回值

所有张量须位于同一设备。Triton 实现要求 NPU 存储；参考实现也接受 CPU 张量，
但导入包仍需已配置的 native runtime、torch_npu、ACL 以及可用的 NPU。

| 参数 | 类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `q` | Tensor | 必填 | 连续的 FP16/BF16 打包查询 `[tokens, Hq, D]`。 |
| `k_cache` | Tensor | 必填 | 连续的 key 缓存 `[pages, P, Hkv, D]`，dtype 与 q 一致。 |
| `v_cache` | Tensor | 必填 | 与 `k_cache` 同形状、同 dtype。 |
| `cu_q_lens` | Tensor | 必填 | 连续 int32 `[B+1]`；从 0 开始、单调不减、末尾等于 `tokens`。实现会读取该值构建调度并校验。 |
| `block_tables` | Tensor | 必填 | 连续 int32/int64 物理页 ID `[B, max_pages]`；每行须覆盖 `ceil(seqlens_kv[b] / P)` 个页。 |
| `seqlens_kv` | Tensor | `None` | 连续 int32 `[B]`，每个请求的总 KV 长度（含其查询行）；`None` 表示 KV 长度等于查询长度。取值须不小于查询长度。 |
| `sinks` | Tensor | `None` | FP16/BF16 `[Hq]` sink logits，dtype 与 query 一致；`None` 关闭 sink。 |
| `gqa_interleave` | bool | `False` | 使用交错的查询头到 KV 头映射。 |
| `softmax_scale` | float | `None` | 有限正数 QK 缩放系数；`None` 取 `D**-0.5`。 |
| `impl` | str | `"auto"` | `"auto"`、`"triton"` 或 `"torch_native"`。 |

其他约束：`Hq` 为正且可被 `Hkv` 整除；`P` 为不小于 16 的 2 的幂；`D` 为
`[16, 208]` 内的整数。该上限在 arch35 上验证；更大的 head_dim 会超出
128 行查询分块 kernel 的单 tile UB 预算。

| 返回值 | 类型 | 形状 | dtype / 设备 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | `[tokens, Hq, D]` | 与 q 一致 | 新分配的结果；`tokens=0` 返回空张量。 |

## triton 后端

- 在 host 上构建 LPT（最长处理时间优先）调度：每个任务对应一组
  （请求、128 行查询分块、查询头），按预计扫描的 KV 块数分配到 cube 核；
  grid 大小等于 cube 核数，因此无需全局原子操作。
- 缓存前缀覆盖的 KV 块不加掩码；其后的块从共享的布尔掩码表读取因果掩码，
  不再逐行重算。sink logit 参与在线最大值与分母的初始化。
- 当 `page_size < 128` 且整除 128 时，页聚合 kernel 把连续 128 行 KV 折进
  一条 cube 指令；否则由 block pointer kernel 每次扫描一个 `BLOCK_SIZE_N`
  分块。
- 除调度输入（每次调用拷贝到 host 一次）外，元数据均在设备上读取。不支持
  NPU Graph 捕获。
- arch35 上的 Triton-Ascend 3.6 支持非 2 的幂 head_dim 与 GQA 分组；arch22
  仍未验证。

## torch_native 后端

- 按 block table 收集每个请求的有效页，按所选映射展开共享 KV 头，并以查询行
  分块计算 FP32 softmax 注意力。
- 将 sink logit 作为额外的分数拼接后丢弃其值列，使其只扩大分母；空查询行保持
  零初始化输出。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill

device = "npu:0"
q = torch.randn(129, 64, 64, dtype=torch.bfloat16, device=device)
k_cache = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
v_cache = torch.randn_like(k_cache)
cu_q_lens = torch.tensor([0, 129], dtype=torch.int32, device=device)
block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
sinks = torch.randn(64, dtype=torch.bfloat16, device=device)

output = paged_attention_sink_prefill(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    cu_q_lens=cu_q_lens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
)
reference = paged_attention_sink_prefill(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    cu_q_lens=cu_q_lens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
    impl="torch_native",
)
torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
```
