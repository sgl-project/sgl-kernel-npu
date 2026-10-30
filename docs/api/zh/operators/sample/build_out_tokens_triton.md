# build_out_tokens_triton

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

默认 `impl="auto"` 与 `triton` 注册到同一入口。注册成功不代表所有芯片均已完成验证。

## 函数原型

```python
from sgl_kernel_npu.operators.sample import build_out_tokens_triton

def build_out_tokens_triton(
    *, draft_tokens, correct_len, bonus, verify_num_draft_tokens, gamma, impl="auto"
):
    ...
```

## 功能说明

将每行 draft token 复制到 `[B, gamma + 1]` 输出，在末尾补 0，并在每行 `correct_len` 指定的位置写入 bonus token。bonus 之后的 draft 元素会保留；调用方需根据接受长度确定已提交的有效前缀。`correct_len` 须在 `[0, gamma]` 范围内。

## 参数与返回值

| 参数 | 类型 | Shape / dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `draft_tokens` | `torch.Tensor` | `[B, gamma]`，int32 或 int64 | 必填 | NPU 上的 draft token ID。 |
| `correct_len` | `torch.Tensor` | `[B]`，int32 或 int64 | 必填 | 每行接受的 draft 长度，与输入位于同一设备。 |
| `bonus` | `torch.Tensor` | `[B]`，int32 或 int64 | 必填 | 每行 bonus token ID，与输入位于同一设备。 |
| `verify_num_draft_tokens` | `int` | — | 必填 | 须等于 `gamma + 1`。 |
| `gamma` | `int` | — | 必填 | 每行 draft token 数，须非负。 |
| `impl` | `str` | — | `"auto"` | 可选 `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape / dtype | 描述 |
| --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, gamma + 1]`，`torch.int64` | 在输入设备上新建的输出，不修改输入。 |

## triton 后端

使用 Triton kernel 构造输出，支持有步长的输入；调用方负责保证接受长度在有效范围内。

## torch_native 后端

使用 Torch 拼接和 scatter 构造相同输出。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import build_out_tokens_triton

drafts = torch.tensor([[10, 11, 12], [20, 21, 22]], device="npu")
lengths = torch.tensor([1, 3], device="npu")
bonus = torch.tensor([99, 98], device="npu")
out = build_out_tokens_triton(draft_tokens=drafts, correct_len=lengths, bonus=bonus,
                       verify_num_draft_tokens=4, gamma=3)
# out 为 [[10, 99, 12, 0], [20, 21, 22, 98]]
# 每行有效前缀的长度为 lengths + 1。
```
