# qsa_prefill

## 芯片支持

arch22 拆成两列，因为该家族的两个成员验证程度不同。A3 指 Ascend910_93 系列，
A2 指 Ascend910B 系列。

| 实现 | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `ascendc` | 支持 | 待验证 | 未实现 |
| `torch_native` | 支持 | 待验证 | 待验证 |
| `triton` | 未实现 | 未实现 | 未实现 |
| `tilelang` | 未实现 | 未实现 | 未实现 |

`ascendc` 实现注册到 arch22，该代际同时包含 A2 与 A3；`SGL_KERNEL_ENABLE_A3_ONLY_OPS`
会为 Ascend950 系列之外的所有 SoC 编译其 kernel。没有任何机制排除 A2：kernel 使用
`__CCE_AICORE__ == 220`（两个家族都是该值），host 守卫只拒绝 Ascend950，下发规模按
`GetCoreNumAic()` 自适应。A2 标为"待验证"而非"不支持"，是因为尚无 A2 实测数据，而
不是已知存在限制。只有 Ascend950 构建不包含原生入口，在其上调用会在
`torch.ops.npu.qsa_prefill_prepare` 抛出 `AttributeError`。

硬件验证使用 Ascend910_9382 与 Ascend910_9362（910C），CANN 9.0.0/9.1.0，
torch 2.10 与 torch_npu 2.10。arch35 未实现。

## 函数原型

默认选项为 `auto`，当前与 `ascendc` 注册到同一个入口函数。不传 `impl` 或显式传入
`"auto"` 时均使用 Ascend C 流水线。

```python
from sgl_kernel_npu.operators.attention import qsa_prefill

def qsa_prefill(
    q,
    k,
    v,
    blocks,
    request_table,
    request_row,
    length,
    base,
    scale=1 / 16,
    *,
    impl="auto",
):
    ...
```

## 功能说明

计算 ratio-4 压缩 QSA prefill 的 BF16 主注意力，head dim 为 256，每个 query 使用
512 个预选逻辑 block。选择结果由调用方提供；本算子不计算 indexer 分数，也不做
Top-K，且不执行任何分布式通信。

对第 $r$ 行 query，可见逻辑位置由每个非负选中 block 的四个 token 位置，加上该行
自身的因果尾部组成：

$$
P_r=\{4b+j \mid b \in \mathrm{blocks}_r,\ b \ge 0,\ 0 \le j < 4,\ 4b+j < L\}
\ \cup\ \{p \mid v_r-(v_r \bmod 4) \le p < v_r,\ 0 \le p < L\},
\quad v_r=\mathrm{base}+r+1
$$

其中 $L$ 为 `length`。选择顺序和重复项都被保留；每个位置通过 request table 映射为
物理槽位 $s=T[\mathrm{request\_row},p]$，当 $s \notin [0,\mathrm{capacity})$ 时丢弃。
设本地 query head $h$ 属于 KV head $\lfloor h/(H/KV) \rfloor$：

$$
Y_{r,h,:}=\sum_{p \in P_r}\mathrm{softmax}_p
\left(\alpha \sum_{d} Q_{r,h,d}K_{s(p),\lfloor h/(H/KV)\rfloor,d}\right)
V_{s(p),\lfloor h/(H/KV)\rfloor,:}
$$

$\alpha$ 为 `scale`。若 $P_r$ 不含任何有效位置，该行输出为 0 而非 NaN。logits、
softmax 与 value 累加使用 float32，最终一次性舍入为 BF16。

## 参数与返回值

公开 operator 在选择实现之前校验下表中的公共约定，因此两个后端拒绝相同的输入。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[R, H, 256]` | `torch.bfloat16` | 必填 | query；连续，位于 NPU 设备。`R == 0` 返回空结果。`(H, KV)` 必须为 `(16, 2)`、`(8, 1)`、`(6, 1)`、`(3, 1)`、`(2, 1)`、`(24, 2)` 或 `(12, 1)`。 |
| `k` | `torch.Tensor` | `[capacity, KV, 256]` | `torch.bfloat16` | 必填 | key cache；连续，与 q 同设备。要求 `capacity < 2**31`，且 `length > 0` 时不可为空。 |
| `v` | `torch.Tensor` | `[capacity, KV, 256]` | `torch.bfloat16` | 必填 | value cache；shape、dtype、设备与 `k` 相同且连续。 |
| `blocks` | `torch.Tensor` | `[R, 512]` | `torch.int32` | 必填 | 每个 query 选中的逻辑 block；连续，与 q 同设备。负值以及位置达到 `length` 的 block 被忽略。 |
| `request_table` | `torch.Tensor` | `[requests, max_length]` | `torch.int32` | 必填 | 逻辑到物理槽位映射表，与 q 同设备；要求 `requests > 0` 且列 stride 为 1。行 stride 可包含 padding。 |
| `request_row` | `torch.Tensor` | `[1]` | `torch.int64` | 必填 | 连续 NPU tensor，保存调用方有效的 `request_table` 行号，仅在设备侧读取。 |
| `length` | `int` | — | — | 必填 | 序列 token 长度；要求 `0 <= length <= request_table.shape[1]`。CPU 元数据，不回读设备。 |
| `base` | `int` | — | — | 必填 | q 第一行之前的逻辑位置，通常为 `length - R`。要求 `-2**31 <= base` 且 `base + R < 2**31`。 |
| `scale` | `float` | — | — | `1 / 16` | logits 乘数；必须为有限正数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"ascendc"`、`"torch_native"`。`auto` 选择 Ascend C。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[R, H, 256]` | `torch.bfloat16` | 新建输出，与 q 同设备。无可见 token 的行为 0；不修改输入。 |

非法元数据在任何 kernel 下发之前抛出 `ValueError`。head 切分与 KV 复制由调用方
负责：当 TP 大于全局 KV head 数时，需为该 rank 传入复制后的 KV head。

对 24 Q / 2 KV 的模型（如 Qwen3.8-Flash-Next），本地布局为 TP1 `(24, 2)`、
TP2 `(12, 1)`、TP4 `(6, 1)`、TP8 `(3, 1)`，四者均支持。对 16 Q / 2 KV 的模型
（如 Whittle-Next），TP1、TP2、TP8 分别为 `(16, 2)`、`(8, 1)`、`(2, 1)`。

## ascendc 后端

- QK、online softmax、PV、rescale 流水线以及物理 run 准备由 Ascend C 实现，注册为
  `torch.ops.npu.qsa_prefill_prepare` 与 `torch.ops.npu.qsa_prefill_runs`。两者是
  内部入口，只能传入本算子生成的元数据，不接受手写 run 描述符。
- query 打包、512 项选择的完整相等性判断以及每 query 的因果尾部合并由 Triton NPU
  kernel 完成，因此该后端同时依赖两套运行时。只有选中 block 列表逐项同序相等的
  精确 16/8/4/2/1 query 分组才共享基础注意力计算。
- 基础输出与 LSE 在尾部合并前保持 float32。结果与其他注意力实现不是逐位一致的。
- 16 个 query 的共享组占用每级 GM workspace 块中 `16 * (H / KV)` 行，该块容纳
  128 行，因此单次下发要求 `H / KV <= 8`。`(16, 2)`、`(8, 1)`、`(6, 1)`、
  `(3, 1)`、`(2, 1)` 满足该条件，单次下发完成。注意力按 query head 独立，所以每个
  KV head 需要 12 个 query 的 `(24, 2)` 与 `(12, 1)` 分别拆成 4 次和 2 次下发
  （每次 6 个 head 的连续切片），
  再按 head 顺序拼接。这些切片需要拷贝，`(24, 2)` 还会为每个 KV head 各拷贝一次，
  因此这两种布局比内核直接支持的布局慢。
- tiling 不可变，按完整 shape、scale、设备以及两个本地 head 数缓存。图捕获之前需
  先预热对应 shape。
- 长序列、共享选择的 prefill 明显快于独立 query 计算；较短或完全不共享的选择可能
  更慢。本算子是显式 API，不是默认分发路径。

## torch_native 后端

- 展开选择、通过 request table 映射，并以 float32 累加计算稠密 softmax 注意力。它
  与 Ascend C 路径不共享代码：没有打包、没有共享分组判断、没有 run 描述符、没有
  独立的尾部合并，因此可以作为精度参考。
- 由于 gather 得到的 key/value 规模按 `R x 2052 x KV x 256` 增长，实现按行分块处理。
  它是参考实现，不是性能路径。
- 归约顺序与 kernel 不同，不保证逐位一致。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.attention import qsa_prefill

rows, heads, kv_heads, length = 17, 16, 2, 8192
q = torch.randn(rows, heads, 256, device="npu", dtype=torch.bfloat16)
k = torch.randn(length, kv_heads, 256, device="npu", dtype=torch.bfloat16)
v = torch.randn_like(k)
blocks = torch.arange(512, device="npu", dtype=torch.int32).repeat(rows, 1)
request_table = torch.arange(length, device="npu", dtype=torch.int32)[None]
request_row = torch.zeros(1, device="npu", dtype=torch.int64)

print(list_impls(qsa_prefill))  # ['ascendc', 'auto', 'torch_native']
args = (q, k, v, blocks, request_table, request_row, length, length - rows)
out = qsa_prefill(*args)  # 默认 auto，使用 Ascend C
out_native = qsa_prefill(*args, impl="torch_native")
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
