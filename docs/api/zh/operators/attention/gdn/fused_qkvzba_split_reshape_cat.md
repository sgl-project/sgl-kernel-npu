# fused_qkvzba_split_reshape_cat

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat

def fused_qkvzba_split_reshape_cat(
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

把 Qwen3-Next 融合投影 `in_proj_qkvz` / `in_proj_ba` 的输出按 head group 拆成 q、k、v、z、b、a，并重排成下游 GDN attention 需要的布局。算子只重排数据，不改变元素值；输出的顺序和形状与输入不同。

源行按 head group 打包，记 `QK = num_heads_qk`、`V = num_heads_v`、`r = V // QK`、`dq = head_qk`、`dv = head_v`：

```text
mixed_qkvz 行 = [ g0: q(dq) | k(dq) | v(r*dv) | z(r*dv) ][ g1: ... ] ...   (QK 组，共 QK*(2*dq+2*r*dv))
mixed_ba   行 = [ g0: b(r) | a(r) ][ g1: ... ] ...                        (共 2*V)
```

输出布局：

```text
mixed_qkv = [ all q | all k | all v ]      宽度 2*QK*dq + V*dv
z         = [rows, V, dv]
b, a      = [rows, V]
```

映射关系（`g ∈ [0,QK)`、`j ∈ [0,r)`、`i ∈ [0,dq)`）：

```text
mixed_qkv[t, g*dq + i]             = mixed_qkvz[t, g*(2*dq+2*r*dv) + i]
mixed_qkv[t, QK*dq + g*dq + i]     = mixed_qkvz[t, g*(2*dq+2*r*dv) + dq + i]
mixed_qkv[t, 2*QK*dq + g*r*dv + m] = mixed_qkvz[t, g*(2*dq+2*r*dv) + 2*dq + m]
z[t, g*r + j]                      = mixed_qkvz[t, g*(2*dq+2*r*dv) + 2*dq + r*dv + j*dv : ... + dv]
b[t, g*r + j]                      = mixed_ba[t, g*2*r + j]
a[t, g*r + j]                      = mixed_ba[t, g*2*r + r + j]
```

## 参数与返回值

| 参数 | 类型 / Shape / Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- |
| `mixed_qkvz` | Tensor `[rows, num_heads_qk*(2*head_qk + 2*(num_heads_v//num_heads_qk)*head_v)]`，浮点 | 必填 | 二维、连续；行宽必须与 head 参数一致。 |
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

**被断言的约束**：两个输入均为二维、行数相同、均连续、`num_heads_v % num_heads_qk == 0`、两个输入行宽分别等于上述公式、`rows > 0`。断言在启动 kernel 前执行，错误信息带实际 shape/stride。

**调用要求**：两个输入需位于同一 NPU。输入不被修改；Triton 后端会分配四个输出。

## triton 后端

要求输入连续且 `rows > 0`；wrapper 会在启动 kernel 前检查。输出张量在对应输入所在设备上新分配。

## torch_native 后端

独立 PyTorch 参考实现使用 reshape、切片和拼接；与 Triton 一样要求输入连续。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat

QK, V, HEAD = 2, 4, 128  # Qwen3-Next TP=8 shard
rows = 1
qkvz = torch.arange(rows * QK * (2 * HEAD + 2 * (V // QK) * HEAD), dtype=torch.float32)
qkvz = qkvz.reshape(rows, -1).to("npu:0")
ba = torch.arange(rows * 2 * V, dtype=torch.float32).reshape(rows, -1).to("npu:0")

mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
    qkvz, ba, QK, V, HEAD, HEAD, impl="triton"
)
# mixed_qkv: [1, 1024]; z: [1, 4, 128]; b/a: [1, 4]
assert torch.equal(mixed_qkv[:, 128:256], qkvz[:, 768:896])  # g1 的 q
assert torch.equal(mixed_qkv[:, 256:384], qkvz[:, 128:256])  # g0 的 k
assert torch.equal(z[:, 3], qkvz[:, 1408:1536])  # g1 的第二个 v head 对应的 z
assert torch.equal(b[:, 2], ba[:, 4])  # g1 的 b0
assert torch.equal(a[:, 0], ba[:, 2])  # g0 的 a0
```
