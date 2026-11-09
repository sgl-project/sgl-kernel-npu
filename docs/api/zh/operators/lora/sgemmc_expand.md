# sgemmc_expand

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.lora import sgemmc_expand

def sgemmc_expand(x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y,
                  *, impl="auto"): ...
```

`impl="auto"` 选择 `ascendc`。使用 PyTorch 实现时请显式传入 `impl="torch_native"`。

## 功能说明

`sgemmv_expand` 的 cube 版本：计算第二个 LoRA 乘积 $h_t B_i^{\top}$ 并累加到
`y`，但 `x` 与输出保持同一激活 dtype，不再使用 float32 中间结果。对 rank 为
$r$ 的适配器 $i$、输出列为 $[o_s, o_{s+1})$ 的切片 $s$：

$$ y[t,\ o_s:o_{s+1}] \mathrel{+}= x[t,\ s r:(s+1)r] \, B_i[o_s:o_{s+1},\ :r]^{\top} $$

其中 $o_s$ = `slice_offsets[s]`。`lora_indices[b] < 0` 或 rank 为 0 的 block 不产生
增量，对应行保持原值；`[slice_offsets[0], slice_offsets[-1])` 之外的列永远不会被
写入。

每个 (token, slice) 乘积由 cube 单元在 float32 中计算，并与已有的 `y` 在 float32
中相加，写回时一次性舍入到 `y` 的 dtype。算子原地修改 `y` 并返回共享同一存储的
Tensor。

## 参数与返回值

所有 Tensor 必须位于同一 NPU。`slice_count = slice_offsets.numel() - 1`，
`max_rank = x.size(1) // slice_count`。

| 参数 | 类型 | Shape | Dtype | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[tokens, slice_count * max_rank]` | `float16 / bfloat16` | 必填 | 连续存储。`tokens > 0`。 |
| `weight` | `torch.Tensor` | `[num_loras, hidden_out, max_rank]` 或 `[num_loras, 1, hidden_out, max_rank]` | 与 `y` 相同 | 必填 | 连续存储。适配器 rank 之后的列不会被读取。 |
| `lora_indices` | `torch.Tensor` | `[num_blocks]` | `int32` | 必填 | 每个 sequence block 的适配器索引，取值为 `-1` 或 `[0, num_loras)`。 |
| `seq_len` | `torch.Tensor` | `[num_blocks]` | `int32` | 必填 | 第 `b` 项是 block `b` 的 token 数。 |
| `lora_ranks` | `torch.Tensor` | `[num_loras]` | `int32` | 必填 | 每个适配器的 rank。已验证 `16`、`32`、`64`；rank 0 跳过该 block。 |
| `slice_offsets` | `torch.Tensor` | `[slice_count + 1]` | `int32` | 必填 | 至少两个元素。必须递增；末项可以小于 `y.size(1)`。 |
| `y` | `torch.Tensor` | `[tokens, hidden_out]` | `float16 / bfloat16` | 必填 | 连续存储，原地累加。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字。`auto`、`ascendc`、`torch_native` |

| 返回 | 类型 | Shape 与 dtype | 说明 |
| --- | --- | --- | --- |
| `sgemmc_expand` | `torch.Tensor` | 同 `y` | 与原地修改的 `y` 共享存储。 |

## ascendc 后端

`lora_indices`、`seq_len`、`lora_ranks` 和 `slice_offsets` 的取值由设备读取，调用方需保证合法。每个切片宽度和 `max_rank` 建议为 16 个元素的倍数，以满足两字节 `DataCopy` 的 32 字节对齐。Kernel 按 `MatmulApiTiling` 选出的 `baseN` 步进拷贝结果，因此切片宽度还应取 2 的幂（或 `baseN` 的整数倍），保证最后一步不越过切片边界。迁移后的 Kernel 直接把 rank 送入 cube matmul，没有 `sgemmv_expand` 的 rank-8 补齐逻辑，因此不支持小于 16 的 rank。

## torch_native 后端

与 `ascendc` 采用相同的计算语义、原地累加和返回别名。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.lora import sgemmc_expand

tokens, slices, max_rank, hidden_out = 4, 1, 16, 4096
x = torch.randn(tokens, slices * max_rank, device="npu", dtype=torch.float16)
weight = torch.randn(2, hidden_out, max_rank, device="npu", dtype=torch.float16) * 0.02
lora_indices = torch.tensor([0, 1, 0, 1], device="npu", dtype=torch.int32)
seq_len = torch.ones(4, device="npu", dtype=torch.int32)
lora_ranks = torch.tensor([16, 16], device="npu", dtype=torch.int32)
slice_offsets = torch.tensor([0, hidden_out], device="npu", dtype=torch.int32)
y = torch.zeros(tokens, hidden_out, device="npu", dtype=torch.float16)

expected = sgemmc_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y.clone(),
    impl="torch_native",
)
result = sgemmc_expand(
    x, weight, lora_indices, seq_len, lora_ranks, slice_offsets, y
)  # auto -> ascendc；原地修改
assert result.data_ptr() == y.data_ptr()
torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
```
