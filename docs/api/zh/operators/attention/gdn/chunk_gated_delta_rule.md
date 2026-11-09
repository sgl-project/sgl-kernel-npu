# chunk_gated_delta_rule

## 芯片支持

| 实现 | 状态 dtype | arch 2201 | arch 3510 |
| --- | --- | --- | --- |
| `ascendc` | BF16 | 支持 | 支持 |
| `ascendc` | FP32 | 不支持 | 支持 |
| `torch_native` | BF16 | 支持 | 支持 |
| `torch_native` | FP32 | 未验证 | 支持 |

状态 dtype 由 `initial_state` 决定，可选 BF16 或 FP32。可选输入 `chunk_state`
必须使用相同 dtype，返回值 `final_state` 也保持该 dtype。
`query`、`key`、`value`、`beta` 和返回值 `out` 均为 BF16；可选输入 `g`
为 FP32，`actual_seq_lengths` 为 int32。

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule

def chunk_gated_delta_rule(
    query,
    key,
    value,
    *,
    beta=None,
    initial_state=None,
    actual_seq_lengths=None,
    scale=None,
    g=None,
    chunk_state=None,
    impl="auto",
):
    ...
```

`auto` 默认使用 `ascendc`；`torch_native` 使用 PyTorch 运算。
原生实现不可用时报错，不自动切换实现。
虽然签名中默认为 `None`，`beta`、`initial_state`、`actual_seq_lengths` 必须提供。

## 功能说明

计算分块 Gated Delta Rule attention，返回每个 token 的输出和各序列的最终状态。
序列按 `actual_seq_lengths` 拼接存储，各序列独立更新状态。

对每个 value 头，状态矩阵 `S` 的形状为 `[Dv, Dk]`，逐 token 计算如下：

```text
D_t = exp(g_t) * S_(t-1)      # g=None 时不衰减
u_t = beta_t * (v_t - D_t @ k_t)
S_t = D_t + outer(u_t, k_t)
out_t = scale * (S_t @ q_t)
```

其中 `outer` 表示向量外积。`scale=None` 时取 `1.0`，显式 `0.0` 有效。
`torch_native` 内部使用 FP32 计算。

## 参数与返回值

`B` 为序列数，`T` 为总 token 数，`Nk`、`Nv` 分别为 query/key 和 value 的头数，
`Dk`、`Dv` 分别为对应的特征维度。

| 参数 | Shape | Dtype / 默认值 | 说明 |
| --- | --- | --- | --- |
| `query`、`key` | `[T, Nk, Dk]` | BF16 | 两者形状相同。 |
| `value` | `[T, Nv, Dv]` | BF16 | Value 张量。 |
| `beta` | `[T, Nv]` | BF16，必填 | Delta 更新系数。 |
| `initial_state` | `[B, Nv, Dv, Dk]` | BF16 / FP32，必填 | 初始状态。FP32 状态详见[芯片支持](#芯片支持)。 |
| `actual_seq_lengths` | `[B]` | int32，必填 | 各序列长度，不是累计边界；`ascendc` 要求每项大于零、总和为 `T`。 |
| `scale` | — | float / `None` | 输出缩放，`None` 时为 `1.0`。 |
| `g` | `[T, Nv]` | FP32 / `None` | 对数衰减，`None` 表示不衰减。 |
| `chunk_state` | `[totalChunks, Nv, Dv, Dk]` | 与初始状态相同 / `None` | 可选块状态缓冲区，`totalChunks = sum(ceil(length / 64))`。 |
| `impl` | — | `"auto"` | 可选 `auto`、`ascendc`、`torch_native`，仅限关键字。 |

| 返回值 | Shape | Dtype |
| --- | --- | --- |
| `out` | `[T, Nv, Dv]` | BF16 |
| `final_state` | `[B, Nv, Dv, Dk]` | 与 `initial_state` 相同 |

使用时请注意：

- `ascendc` 的所有张量必须位于同一 NPU；输入可非连续，内部转为连续工作张量。
  `torch_native` 要求张量位于同一设备。
- 各维度为正；`Nk, Nv <= 64`、`Dk, Dv <= 128`，且 `Nv % Nk == 0`。
  Value 头 `h` 对应 query/key 头 `h // (Nv // Nk)`。
- 调用方保证长度内容和块缓冲区容量有效，原生实现不检查这两项。
  `torch_native` 也接受零长度序列，其最终状态等于对应初始状态。
- `initial_state` 不被修改。连续的 `chunk_state` 原地写入各块的进入状态，
  先按序列、再按块排序；非连续缓冲区使用临时副本，不回写原张量。
  函数只返回 `(out, final_state)`。
