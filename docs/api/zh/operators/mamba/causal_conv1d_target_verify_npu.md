# causal_conv1d_target_verify_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## 函数原型

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

def causal_conv1d_target_verify_npu(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    retrieve_parent_token=None,
    pad_slot_id=-1,
    validate_data=False,
    *,
    impl="auto",
):
    ...
```

## 功能说明

为 speculative target verify 计算逐通道因果卷积，可选 bias 和 SiLU 激活。每个有效候选的卷积后窗口按 `intermediate_state_indices` 写入 scratch。被接受的窗口之后由调用方提交，因此已提交的 `conv_state` 保持只读。

线性链按 token 顺序消费输入；tree verify 则根据 `retrieve_next_token` 和 `retrieve_next_sibling` 遍历，并将推导出的父节点映射写入 `retrieve_parent_token`。

## 参数

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, D]` 或 `[B, D, S]` | 浮点；输出使用 `x.dtype`，wrapper 未列举受支持 dtype 集合 | 必填 | 候选 token；二维输入按 `S = 1` 处理，`B`、`S` 必须非空。 |
| `conv_state` | `torch.Tensor` | `[cache_lines, D, state_len >= K - 1]` | 浮点；历史值使用该 tensor 的元素 dtype | 必填 | 已提交卷积状态；只读。 |
| `weight` | `torch.Tensor` | `[D, K]` 或 `[K, D]` | 浮点；加载后参与 FP32 计算 | 必填 | 逐通道卷积权重；`K` 必须在 `2..4`。 |
| `bias` | `torch.Tensor` 或 `None` | `[D]` 或 — | 浮点；加载后参与 FP32 计算 | `None` | 可选逐通道 bias。 |
| `activation` | `bool`、`str` 或 `None` | — | — | `None` | `True` 映射为 `"silu"`，`False` 映射为 `None`；字符串可选 `"silu"`、`"swish"`。 |
| `conv_state_indices` | `torch.Tensor` | 一维，`[>= B]` | 整数；未显式检查 dtype | `None`，但必须提供 | 为每个 batch 项选择已提交状态行；等于 `pad_slot_id` 的项使该候选失效。 |
| `intermediate_conv_window` | `torch.Tensor` | `[scratch_size, steps >= S, D, window >= K - 1]` | 浮点；未列举受支持 dtype 集合 | `None`，但必须提供 | 有效候选的 scratch 窗口，原地更新。 |
| `intermediate_state_indices` | `torch.Tensor` | 一维，`[>= B]` | 整数；未显式检查 dtype | `None`，但必须提供 | 为每个 batch 项选择 scratch 行。 |
| `retrieve_next_token` | `torch.Tensor` 或 `None` | `[rows >= B, steps >= S]` 或 — | 整数 tree 映射 | `None` | 提供后启用 tree 模式，此时三个 `retrieve_*` 参数必须全部提供。 |
| `retrieve_next_sibling` | `torch.Tensor` 或 `None` | `[rows >= B, steps >= S]` 或 — | 整数 tree 映射 | `None` | tree 模式的下一兄弟节点映射。 |
| `retrieve_parent_token` | `torch.Tensor` 或 `None` | `[rows >= B, steps >= S]` 或 — | 整数 tree 映射 | `None` | 在 tree 模式接收推导出的父节点映射；有效候选会原地更新。 |
| `pad_slot_id` | `int` | — | — | `-1` | 标识失效状态 slot 的哨兵值。 |
| `validate_data` | `bool` | — | — | `False` | 启用额外 stride 检查；不是数值正确性验证模式。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`。 |

应在同一 NPU 上使用相互兼容的浮点 tensor。实现不全面检查 dtype 组合，本文也不宣称未实测组合得到支持。

## 返回值与原地状态

返回与 `x` 具有相同逻辑 shape、dtype 和设备的 tensor；二维输入得到二维输出。

- **线性模式**（`retrieve_next_token is None`）：返回 tensor 与 `x` 共用存储，token 被消费时 `x` 会被覆盖。`retrieve_next_sibling`、`retrieve_parent_token` 也必须为 `None`。
- **tree 模式**（三个 `retrieve_*` tensor 全部提供）：通过 `empty_like(x)` 新建输出，不修改 `x`，并更新 `retrieve_parent_token`。
- 两种模式都会更新 `intermediate_conv_window`。`conv_state`、`weight`、可选 `bias`、`conv_state_indices`、`intermediate_state_indices`、`retrieve_next_token` 和 `retrieve_next_sibling` 只读；尤其 `conv_state` 不会被本算子修改。

## triton 后端

- `conv_state_indices`、`intermediate_conv_window`、`intermediate_state_indices` 必须非 `None`；支持宽度 2 到 4，以及两种权重布局。
- `validate_data=True` 时，额外要求 `x`、`conv_state` 的通道维连续（`stride(1) == 1`），`[D, K]` 权重的宽度维连续，或 `[K, D]` 权重的通道维连续。一般调用仍须保证索引范围和布局有效。
- `auto` 是 `triton` 的别名，不进行动态选择或 fallback。导入 registry 仍会执行 ACL 架构检测。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

b, d, s, k = 1, 4, 2, 3
x = torch.randn(b, d, s, device="npu", dtype=torch.bfloat16)
conv_state = torch.randn(1, d, k - 1, device="npu", dtype=torch.bfloat16)
weight = torch.randn(d, k, device="npu", dtype=torch.bfloat16)
windows = torch.zeros(1, s, d, k - 1, device="npu", dtype=torch.bfloat16)

out = causal_conv1d_target_verify_npu(
    x,
    conv_state,
    weight,
    activation="silu",
    conv_state_indices=torch.tensor([0], device="npu", dtype=torch.int64),
    intermediate_conv_window=windows,
    intermediate_state_indices=torch.tensor([0], device="npu", dtype=torch.int64),
)
# 线性模式：out 与 x 共用存储，windows 保存 scratch 检查点。
```
