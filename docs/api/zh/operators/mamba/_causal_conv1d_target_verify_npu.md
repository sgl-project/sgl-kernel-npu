# causal_conv1d_target_verify_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数原型

```python
from typing import Optional, Union
import torch
from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

def causal_conv1d_target_verify_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    validate_data: bool = False,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` 固定选择 `triton`，省略 impl 效果相同；不执行运行时后端回退。

## 功能说明

设卷积宽度为 K，历史按旧到新排列，历史 h 来自 committed state 或祖先候选：

$$u_{t,d}=b_d+\sum_{j=0}^{K-2}w_{d,j}h_{t,d,j}+w_{d,K-1}x_{t,d},\qquad y_{t,d}=\mathrm{act}(u_{t,d}).$$

无 bias 时 b=0；act 为恒等或 SiLU（`u*sigmoid(u)`）。FP32 计算后转回 x.dtype。scratch 保存包含当前**原始输入**的后续 K-1 项历史，额外窗口列保持不变。线性模式原地覆盖 x 并返回别名；tree 模式保留 x、返回新张量，并根据 child/sibling 推导和修改 parent。两种模式都修改 scratch，均不修改 committed conv_state，接受后的状态由调用方另行提交。

## 参数与返回值

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
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；`auto`, `triton`, `torch_native` |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 与 x 相同 | `x.dtype` | 线性模式别名；tree 模式新张量。 |

所有 tensor 位于同一 NPU。上游保证索引值域、容量与读入历史有效，避免原地状态的并发写冲突；wrapper 不完整检查所有值域。

## triton 后端

支持宽度 2..4、两种 weight 布局和带 stride 的视图。两个 weight 轴均匹配通道时优先 [D,K]。validate_data=True 要求 x/state 通道 stride=1、[D,K] 权重宽度 stride=1 或 [K,D] 权重通道 stride=1。三个 state/scratch 参数虽然默认 None，实际调用必须提供。

## torch_native 后端

独立 CPU 数值参考：输入 detach 并转 CPU，用 FP32 循环计算，再把结果和 scratch/parent 写回原设备。保留线性别名和 tree 不修改 x 的语义。其耗时不代表高性能 NPU PyTorch 基线。

公共 registry 导入时检测 NPU 架构，`torch_native` 入口也需要该环境。支持范围为已验收的 A3 用例；架构注册不代表所有芯片、shape 或 dtype 均已验证。

## 调用示例

需已安装 sgl-kernel-npu、torch_npu 和 NPU 运行环境。本例显式选择 `torch_native`。 也可选择 `triton` 或 `auto`；比较后端时需隔离各自的可变状态。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu
b, d, s, k = (1, 4, 2, 3)
x = torch.randn(b, d, s, device='npu', dtype=torch.bfloat16)
conv_state = torch.randn(1, d, k - 1, device='npu', dtype=torch.bfloat16)
weight = torch.randn(d, k, device='npu', dtype=torch.bfloat16)
windows = torch.zeros(1, s, d, k - 1, device='npu', dtype=torch.bfloat16)
out = causal_conv1d_target_verify_npu(
    x,
    conv_state,
    weight,
    activation='silu',
    conv_state_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    intermediate_conv_window=windows,
    intermediate_state_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    impl='torch_native',
)
```
