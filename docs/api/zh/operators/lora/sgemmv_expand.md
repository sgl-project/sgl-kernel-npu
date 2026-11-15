# sgemmv_expand

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.lora import sgemmv_expand

def sgemmv_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y, *, impl="auto"): ...
```

`impl="auto"` 选择 `ascendc`。使用 PyTorch 实现时请显式传入 `impl="torch_native"`。

## 功能说明

LoRA 用低秩分解近似权重增量。对基础权重 $W$，适配器 $i$ 学习
$\Delta W_i = B_i A_i$，该层输出为：

$$ y_t = x_t W^{\top} + \frac{\alpha_i}{r_i} x_t A_i^{\top} B_i^{\top} $$

`sgemmv_expand` 负责其中的第二步。输入的 `x` 是 `sgemmv_shrink` 已经算好的低秩中间结果
$h_t = \frac{\alpha_i}{r_i} x_t A_i^{\top}$（缩放已折叠进 $h$），`weight` 是 $B_i$。算子计算
$h_t B_i^{\top}$ 并原地累加到基础层输出 `y`；调用前 `y` 可以是基础 GEMM 的结果，也可以是全零
（只计算 LoRA 分支）。

多 LoRA 场景下，`seq_len[b]` 个连续 token 组成一个 sequence block，同一 block 共用一个
适配器：`lora_indices[b]` 选择适配器，`lora_ranks[i]` 给出其 rank。`lora_indices[b] < 0`
或 rank 为 0 的 block 不产生增量，对应 token 的 `y` 保持不变。

一层输出可能由多个子投影拼接而成（例如 fused QKV 或 gate/up），`slice_offsets` 把它切成
`slice_count` 个输出切片。对 rank 为 `r` 的适配器，切片 `s` 读取 `x` 中第 `s` 段 rank
大小的输入，并写回对应的输出列：

$$ y[t,\ o_s:o_{s+1}] \mathrel{+}= x[t,\ s r:(s+1)r] \, B_i[o_s:o_{s+1},\ :r]^{\top} $$

其中 `o_s = slice_offsets[s]`；`weight` 中 rank `r` 之后的列不读取。累加在 float32 中完成，
写回时一次性舍入到 `y` 的 dtype。函数原地修改 `y` 并返回同一存储的 Tensor；`x`、`weight`
和四个索引张量不变。

## 参数与返回值

所有 Tensor 必须位于同一 NPU。注意`slice_count = slice_offsets.numel() - 1`，`max_rank = x.size(1) // slice_count`。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, slice_count * max_rank]` | `float32` | 必填 | 连续存储。`tokens > 0`。 |
| `weight` | `torch.Tensor` | `[num_loras, hidden_out, max_rank]` 或 `[num_loras, 1, hidden_out, max_rank]` | `float16 / bfloat16` | 必填 | 连续存储，dtype 必须与 `y` 相同。 |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | 必填 | 每个 sequence block 的适配器索引，取值为 `-1` 或 `[0, num_loras)`。按 block 索引，不按 token 索引。 |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | 必填 | 第 `b` 项是 block `b` 的 token 数。 |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | 必填 | 每个适配器的 rank，仅支持 `0`、`8`、`16`、`32`、`64`。 |
| `slice_offsets` | `torch.Tensor` | `[slice_count + 1]` | `int32` | 必填 | 至少两个元素。必须递增，各切片不重叠，末项不超过 `y.size(1)`。 |
| `y` | `torch.Tensor` | `[tokens, hidden_out]` | `float16 / bfloat16` | 必填 | 连续存储，原地累加。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字。`auto`、`ascendc`、`torch_native` |

| 返回 | 类型 | Shape 与 dtype | 说明 |
| --- | --- | --- | --- |
| `sgemmv_expand` | `torch.Tensor` | 同 `y` | 与原地修改的 `y` 共享存储。 |

## ascendc 后端

`lora_indices`、`seq_len`、`lora_ranks` 和 `slice_offsets` 的取值由设备读取，调用方需保证合法。当 `y` 为 float16/bfloat16 时，`slice_offsets` 的每一项必须是 16 个元素的倍数，以满足底层 `DataCopy` 的对齐要求。

## torch_native 后端

与 `ascendc` 采用相同的计算语义、原地写回和返回别名。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmv_expand

tokens, slices, max_rank, hidden_out = 4, 1, 16, 4096
x = torch.randn(tokens, slices * max_rank, device="npu", dtype=torch.float32)
weight = torch.randn(2, hidden_out, max_rank, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
slice_offsets = torch.tensor([0, hidden_out], device="npu", dtype=torch.int32)
y = torch.zeros(tokens, hidden_out, device="npu", dtype=torch.float16)

expected = sgemmv_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y.clone(),
    impl="torch_native",
)
result = sgemmv_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
)  # auto -> ascendc；原地修改
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
```
