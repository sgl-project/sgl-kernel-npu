# fused_qkvzba_split_reshape_cat_contiguous

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat_contiguous

def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

`auto` 与 `triton` 指向同一实现；`torch_native` 是独立的 Torch 参考实现。后端延迟导入，不执行自动回退。

## 功能说明

把 Qwen3.5 分段拼接的投影输出 `[all q | all k | all v | all z]` 与 `[all b | all a]` 拆成 q/k/v、z、b、a，输出布局与 `fused_qkvzba_split_reshape_cat` 完全一致；区别只有输入的行内排布（本算子不按 head group 交错）。

记 `QK = num_heads_qk`、`V = num_heads_v`、`dq = head_qk`、`dv = head_v`：

```text
mixed_qkvz 行 = [ all q (QK*dq) | all k (QK*dq) | all v (V*dv) | all z (V*dv) ]
mixed_ba   行 = [ all b (V) | all a (V) ]
```

输出布局：

```text
mixed_qkv = [ all q | all k | all v ]      宽度 2*QK*dq + V*dv
z         = [rows, V, dv]
b, a      = [rows, V]
```

映射关系：

```text
mixed_qkv[t, 0 : 2*QK*dq + V*dv] = mixed_qkvz[t, 0 : 2*QK*dq + V*dv]
z[t, h]                          = mixed_qkvz[t, 2*QK*dq + V*dv + h*dv : ... + dv]
b[t, h]                          = mixed_ba[t, h]
a[t, h]                          = mixed_ba[t, V + h]
```

算子只重排数据，不改变元素值；输出形状与输入不同。

## 参数与返回值

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `mixed_qkvz` | Tensor `[rows, 2*num_heads_qk*head_qk + 2*num_heads_v*head_v]`，浮点 | 必填 | 二维、连续；行宽必须等于 `2*num_heads_qk*head_qk + 2*num_heads_v*head_v`。 |
| `mixed_ba` | Tensor `[rows, 2*num_heads_v]`，浮点 | 必填 | 二维、连续；dtype 可与 `mixed_qkvz` 不同。 |
| `num_heads_qk` | `int` | 必填 | q/k 头数，必须为正数。 |
| `num_heads_v` | `int` | 必填 | v 头数，必须为正数且能被 `num_heads_qk` 整除。 |
| `head_qk` | `int` | 必填 | q/k 单头维度，必须为正数。 |
| `head_v` | `int` | 必填 | v/z 单头维度，必须为正数。 |
| `impl` | `str` | `"auto"` | 仅限关键字；`auto`、`triton`、`torch_native`。 |

| 返回值 | Shape / Dtype | 含义 |
| --- | --- | --- |
| `mixed_qkv` | `[rows, 2*num_heads_qk*head_qk + num_heads_v*head_v]`，`mixed_qkvz.dtype` | 拼接后的 q/k/v。 |
| `z` | `[rows, num_heads_v, head_v]`，`mixed_qkvz.dtype` | 按头展开的 z。 |
| `b` | `[rows, num_heads_v]`，`mixed_ba.dtype` | b。 |
| `a` | `[rows, num_heads_v]`，`mixed_ba.dtype` | a。 |

**被断言的约束**：两个输入均为二维、行数相同、均连续、`num_heads_v % num_heads_qk == 0`、两个输入行宽分别匹配上述公式。断言在启动 kernel 前执行，错误信息带实际 shape/stride。

**调用要求**：两个输入需位于同一 NPU。输入不被修改。Triton 会分配全部输出；torch_native 返回的 `z` 是 `mixed_qkvz` 的视图，其他输出会被物化。

## triton 后端

要求输入连续；wrapper 会在启动 kernel 前检查。支持空 batch，并返回形状正确的空张量：`[0, width]`、`[0, V, dv]`、`[0, V]`、`[0, V]`。

## torch_native 后端

独立 PyTorch 参考实现按列切片并 reshape、拼接。其 `z` 是 `mixed_qkvz` 的视图；`mixed_qkv`、`b`、`a` 是物化后的输出。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat_contiguous

QK, V, HEAD = 2, 4, 128  # Qwen3.5 TP=8 shard
rows = 1
width = 2 * QK * HEAD + 2 * V * HEAD
qkvz = torch.arange(rows * width, dtype=torch.float32).reshape(rows, -1).to("npu:0")
ba = torch.arange(rows * 2 * V, dtype=torch.float32).reshape(rows, -1).to("npu:0")

mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
    qkvz, ba, QK, V, HEAD, HEAD, impl="triton"
)
# mixed_qkv: [1, 1024]; z: [1, 4, 128]; b/a: [1, 4]
assert torch.equal(mixed_qkv, qkvz[:, : 2 * QK * HEAD + V * HEAD])
assert torch.equal(z, qkvz[:, 2 * QK * HEAD + V * HEAD :].reshape(rows, V, HEAD))
assert torch.equal(b, ba[:, :V])
assert torch.equal(a, ba[:, V:])
```
