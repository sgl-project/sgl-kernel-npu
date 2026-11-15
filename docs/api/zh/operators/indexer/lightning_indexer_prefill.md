# lightning_indexer_prefill

## 硬件支持

arch22 分列，因为该系列的两个成员是分别实测的。A3 指 Ascend910_93 系列，A2
指 Ascend910B 系列。

| 实现 | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `torch_npu` | 支持 | 支持 | 未实现 |
| `torch_native` | 支持 | 支持 | 未验证 |

`torch_npu` 与 `auto` 仅在 arch22 上注册，因为分组下发几何只在该系列上实测过。
正确性在 Ascend910_9362（A3，CANN 9.0）与 Ascend910B3（A2，CANN 9.1.0，
torch 2.10.0，torch_npu 2.10.0.post4）上验证。arch35 未实现：在该代上
`impl="auto"` 抛 `UnsupportedGenerationError`，调用方保留自己的回退路径。

## 接口

```python
import torch
from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

def lightning_indexer_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    sequence_length: int,
    group4: bool = True,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

`auto` 选择 `torch_npu`，省略 impl 等效。无运行时后端回退。

## 功能说明

为单个 ratio-4 因果 prefill 请求的每个 query 选出有序的 Top-512 压缩 key。
打分使用单位 head 权重：

```text
score[i, j] = sum_h relu(q[i, h] . k[j])
```

第 `i` 行对应逻辑位置 `sequence_length - rows + i`，因此对它因果可见的压缩
key 数为 `floor((sequence_length - rows + i + 1) / 4)`。超出该视野的 key 永不
被选中；视野为空的行全部返回 -1。

在 ratio-4 因果 prefill 中，最多四个相邻 query 共享同一段可见 key 范围。
`group4=True` 时这些 query 被批进一次 CANN LightningIndexer 下发的 sequence
轴，而不是为每个 query 复制一行 page table。每个 query 仍各自计算分数与各自
的有序 Top-K，不共享任何选择结果。`group4=False` 让每个 query 独占一个 batch
条目，是对照路径。分组另需 `rows >= 1024` 且 `keys >= 512`；低于任一阈值时两
种设置下发相同的工作量。

本算子不支持任意 mask、学习得到的 head 权重、其他形状、decode 与投机执行。
每个请求调用一次；返回的索引是相对 `k` 的局部索引。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[rows, 4, 128]` | `torch.bfloat16` | 必填 | 该请求最后 `rows` 个 query，连续，位于 NPU 上。 |
| `k` | `torch.Tensor` | `[sequence_length // 4, 1, 128]` | `torch.bfloat16` | 必填 | 压缩 key，连续，与 q 同设备。 |
| `sequence_length` | `int` | — | — | 必填 | 仅关键字。请求 token 长度，取值在 `[0, 2**31)` 的宿主整数。需满足 `rows <= sequence_length` 且 `k.shape[0] == sequence_length // 4`。不回读任何设备值用于判定或相位计算。 |
| `group4` | `bool` | — | — | `True` | 仅关键字。选择分组或独立的下发几何。两者选中相同的 key，因此这是对照开关而非调优旋钮。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；`auto`、`torch_npu`、`torch_native` |

| 返回值 | 类型 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[rows, 512]` | `torch.int32` | 新分配的有序 key 索引，相对 `k`，以 -1 填充。第 `i` 行含 `min(512, floor((sequence_length - rows + i + 1) / 4))` 个有效项。输入不被修改。 |

不支持自动求导：带 `requires_grad` 的输入会被拒绝。不符合契约的输入在任何
kernel 下发之前抛 `ValueError`。

## 并列项的可复现性

精确的并列顺序不属于契约。两个后端以不同顺序累加 128 元素点积与四头求和，
因此分数相差几个 float32 ulp 以内的 key 可能以任意顺序返回；恰好跨越 Top-512
切分边界的并列，还可能让两侧落到不同的 key 上。明确定义、也是测试所断言的
是：

- 选中分数的多重集，
- 每行有效项的个数，
- 返回 key 按分数单调不增的顺序。

而在 `group4=True` 与 `group4=False` 之间，有序索引**是**逐位相同的，包括并列
与填充 —— 这正是四 query 批处理必须保持的不变量，测试套件在全零输入（所有
分数并列）下检查它。

## torch_npu 后端

将请求适配到 `torch_npu.npu_lightning_indexer`，`layout_query="BSND"`、
`layout_key="PA_BSND"`。key 补齐到整页 64 项，并通过恒等 block table 寻址。
分组相位为 `(sequence_length - rows + 1) % 4`，它让每组四个 query 中所有真实
query 共享同一段可见 key 范围；首尾的哑 query 被丢弃。CANN 要求长度元数据中
至少有一个 key，因此可见 key 为零的行在之后被掩回 -1。

## torch_native 后端

独立参考实现，不调用任何 CANN indexer 算子。以 float32 对每个 query 与每个
key 打分，掩掉因果视野外的部分，再做稳定降序排序，因此并列向较小 key 索引
收敛。由于每头点积规模为 `rows * 4 * keys`，按行分块处理。它是参考实现，
不是性能后端。

## 性能

每个 shape 三次预热、8 至 20 次交替顺序的 NPU event 采样；两列均包含 Python
适配、补齐、表构建与掩码。输入为随机非零 BF16 张量。

Ascend910_9362（910C，CANN 9.0）：

| Query 行数 / key 数 | `group4=False` | `group4=True` | 比值 |
| --- | ---: | ---: | ---: |
| 128 / 1024 | 0.447 ms | 0.445 ms | 1.00x |
| 1024 / 512 | 0.608 ms | 0.574 ms | 1.06x |
| 16384 / 4096 | 11.831 ms | 4.732 ms | 2.50x |
| 16384 / 8192 | 18.751 ms | 7.834 ms | 2.39x |
| 8313 / 11102 | 13.831 ms | 6.072 ms | 2.28x |

Ascend910B3（A2，CANN 9.1.0，5 AI core 虚拟化切片）：

| Query 行数 / key 数 | `group4=False` | `group4=True` | 比值 |
| --- | ---: | ---: | ---: |
| 128 / 1024 | 0.721 ms | 0.726 ms | 1.00x |
| 1024 / 512 | 1.319 ms | 1.022 ms | 1.29x |
| 16384 / 4096 | 31.201 ms | 13.184 ms | 2.37x |
| 16384 / 8192 | 65.601 ms | 27.390 ms | 2.39x |
| 8313 / 11102 | 49.188 ms | 21.025 ms | 2.34x |

低于分组阈值的 shape 在两列中下发相同工作量，其差异属于测量噪声。可用
`benchmark/indexer/bench_lightning_indexer_prefill.py` 复现。

## 示例

需要 sgl-kernel-npu、torch_npu 与 NPU 运行时。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

sequence_length, rows = 4096, 1024
q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(sequence_length // 4, 1, 128, device="npu", dtype=torch.bfloat16)
indices = lightning_indexer_prefill(q, k, sequence_length=sequence_length)
print(indices.shape, indices.dtype)  # torch.Size([1024, 512]) torch.int32
```
