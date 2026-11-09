# sgemmv_shrink

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.lora import sgemmv_shrink

def sgemmv_shrink(x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y,
                  slice_count=1, *, impl="auto"): ...
```

`impl="auto"` 选择 `ascendc`。使用 PyTorch 实现时请显式传入 `impl="torch_native"`。

## 功能说明

LoRA 用低秩分解近似权重增量。对基础权重 $W$，适配器 $i$ 学习
$\Delta W_i = B_i A_i$，该层输出为：

$$ y_t = x_t W^{\top} + \frac{\alpha_i}{r_i} x_t A_i^{\top} B_i^{\top} $$

`sgemmv_shrink` 负责其中第一个 LoRA 乘积。`x` 是层输入，`weight` 是 $A_i$，
`lora_scales[i]` 是折叠后的缩放系数 $\alpha_i / r_i$。对某个 sequence block 中
适配器 $i$（rank 为 $r$）的每个 token $t$，切片 $s$ 计算：

$$ h[t,\ s r:(s+1)r] = \lambda_i \, x[t,\ :] \, A_i[s r:(s+1)r,\ :]^{\top} $$

其中 $\lambda_i$ = `lora_scales[i]`，每个适配器的权重块按切片首尾相接存放，即
`weight[i, s*r:(s+1)*r, :]`。`lora_indices[b] < 0` 或 rank 为 0 的 block 不写入，
对应行保持原值。所有点积和缩放都在 float32 中完成，`y` 本身也是 float32，
不涉及舍入。

算子只写入每行被 `slices * rank` 覆盖的列，并返回与 `y` 共享存储的 Tensor；
其余列保持调用前的值。`x`、`weight` 和各索引张量不变。输出可作为
`sgemmv_expand` 的输入。

## 参数与返回值

所有 Tensor 必须位于同一 NPU。`max_rank = y.size(1) // slice_count`。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, hidden_in]` | `float16 / bfloat16` | 必填 | 连续存储。`tokens > 0`，`hidden_in > y.size(1)`。 |
| `weight` | `torch.Tensor` | `[num_loras, slices * max_rank, hidden_in]` 或 `[num_loras, 1, slices * max_rank, hidden_in]` | 与 `x` 相同 | 必填 | 连续存储。适配器 `i` 的切片 `s` 占第 `s*rank_i : (s+1)*rank_i` 行。 |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | 必填 | 每个 sequence block 的适配器索引，取值为 `-1` 或 `[0, num_loras)`。 |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | 必填 | 第 `b` 项是 block `b` 的 token 数。 |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | 必填 | 每个适配器的 rank，仅支持 `0`、`8`、`16`、`32`、`64`。 |
| `lora_scales` | `torch.Tensor` | `[num_loras]` | `float16` | 必填 | 作用于每个计算列的缩放系数。 |
| `y` | `torch.Tensor` | `[tokens, slices * max_rank]` | `float32` | 必填 | 连续存储，原地写入。 |
| `slice_count` | `int` | — | — | `1` | 融合的子投影数量（SGLang 中为 1、2 或 3）。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字。`auto`、`ascendc`、`torch_native` |

| 返回 | 类型 | Shape 与 dtype | 说明 |
| --- | --- | --- | --- |
| `sgemmv_shrink` | `torch.Tensor` | 同 `y` | 与原地修改的 `y` 共享存储。 |

## ascendc 后端

`lora_indices`、`seq_len`、`lora_ranks` 和 `lora_scales` 的取值由设备读取，调用方需保证合法。`CopyOut` 每次搬运 `rank` 个 float32，因此支持的 rank（8/16/32/64）天然满足 `DataCopy` 的 32 字节对齐；`hidden_in` 应为 64 的倍数，这样同时满足 half/bfloat16 输入拷贝对齐和 tile 内 64 元素 float32 `ReduceSum` 的要求。Kernel 按 11776 元素的 tile 遍历 `hidden_in`，并处理末尾不足一个 tile 的部分。

## torch_native 后端

与 `ascendc` 采用相同的计算语义、原地写回和返回别名。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmv_shrink

tokens, slices, max_rank, hidden_in = 4, 1, 16, 2048
x = torch.randn(tokens, hidden_in, device="npu", dtype=torch.float16)
weight = torch.randn(2, slices * max_rank, hidden_in, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
lora_scales = torch.tensor([1.0, 0.5], device="npu", dtype=torch.float16)
y = torch.zeros(tokens, slices * max_rank, device="npu", dtype=torch.float32)

expected = sgemmv_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y.clone(),
    impl="torch_native",
)
result = sgemmv_shrink(
    x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y
)  # auto -> ascendc；原地修改
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
```
