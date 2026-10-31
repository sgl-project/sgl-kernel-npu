# chunk_gated_delta_rule

## 签名与实现选择

```python
from sgl_kernel_npu.operators.attention import chunk_gated_delta_rule

out, final_state = chunk_gated_delta_rule(
    query, key, value, beta=beta, initial_state=initial_state,
    actual_seq_lengths=actual_seq_lengths, impl="auto"
)
```

可选 `g`、`chunk_state` 默认为 `None`，`impl` 默认为 `"auto"`。
虽然 Python 默认值为 `None`，`beta`、`initial_state`、`actual_seq_lengths`
仍必须提供。`auto` 选择 `ascendc`，调用独立的新绑定
`torch.ops.npu.sgl_chunk_gated_delta_rule`。缺少绑定或执行失败会直接报错，
不会静默切换后端。旧 `torch.ops.npu.chunk_gated_delta_rule` 和旧 Python /
Triton 路线保持原样。显式选择 `impl="torch_native"` 使用独立 PyTorch 参考。

省略 `scale` 或传入 `None` 的有效值为 **1.0**；显式数值包括 **0.0** 都保留。

## 参数、输出与副作用

`ascendc` 的所有张量必须在同一 NPU。内部使用连续工作张量，调用方无需改变布局。

| 参数 | shape 与 dtype |
| --- | --- |
| `query`、`key` | BF16 `[T, Nk, Dk]`，形状相同 |
| `value` | BF16 `[T, Nv, Dv]` |
| `beta` | 必填 BF16 `[T, Nv]`，增量更新系数 |
| `initial_state` | 必填 BF16 或 FP32 `[B, Nv, Dv, Dk]`；原生 FP32 状态仅限 arch35 / Ascend950 |
| `actual_seq_lengths` | 必填 int32 `[B]`，按顺序描述打包序列；原生可用范围要求每条序列长度严格大于零且总和等于 `T` |
| `g` | 可选 FP32 `[T, Nv]`，对数衰减；`None` 表示不衰减 |
| `scale` | 可选标量，仅缩放注意力输出 |
| `chunk_state` | 可选 `[totalChunks, Nv, Dv, Dk]`，dtype 与初始状态相同；调用方提供 `sum(ceil(length / 64))` 个块所需容量 |
| `impl` | `auto`、`ascendc`、`torch_native` |
| `out` | BF16 `[T, Nv, Dv]` |
| `final_state` | `[B, Nv, Dv, Dk]`，dtype 与初始状态相同 |

各维度为正；`Nk, Nv <= 64`，`Dk, Dv <= 128`，`Nv % Nk == 0`。
value 第 `h` 个头使用 query/key 的 `h // (Nv // Nk)` 头。
各打包序列使用独立的初始和最终状态。原生校验保留旧版 chunk 尾部维度检查，
不会据此证明首维容量或长度内容合法；调用方负责提供合法长度和足够存储。
独立参考会显式检查长度非负且总和正确，这不代表原生校验已经读取并检查这些内容。

零长度序列不在原生可用范围内。adapter 分配 `final_state` 时不初始化，继承的
arch22/arch35 Kernel 循环对零长度序列跳过所有状态写入，因此对应序列的
`final_state` 未被写入，结果未定义。原生校验不会拒绝这种情况。
独立 `torch_native` 参考接受零长度，并将对应 `initial_state` 作为数学上的最终状态
返回；这一行为不代表原生实现支持零长度序列。

`initial_state` 不被原地修改。**连续** `chunk_state` 写入各块的**进入状态**，
先按序列、再按块排序。**非连续** `chunk_state` 转为工作副本，**不回写原张量**，
严格保留旧 Host 的行为。函数只返回 `(out, final_state)`，不返回临时 chunk 副本。

## 数学定义与独立参考

对每个 value 头，状态 `S` 的方向是 `Dv × Dk`。逐 token 的定义为：

```text
D_t = exp(g_t) S_(t-1)             （没有 g 时 D_t = S_(t-1)）
u_t = beta_t (v_t - D_t k_t)
S_t = D_t + u_t k_t^T
out_t = scale S_t q_t
```

参考以 64 token 分块，使用 FP32 工作计算。块内 `G = cumsum(g)`，因果衰减矩阵
在 `j <= i` 时为 `E_ij = exp(G_i - G_j)`。丢弃的上三角在指数计算**之前**掩码，
不裁剪有效下三角。构造单位下三角系统
`A U = beta * (V - exp(G) K S^T)`，其中
`A = I + tril(beta_i * (K K^T)_ij * E_ij, -1)`，使用 PyTorch 基础运算逐行前代求解，
然后计算因果输出和离开状态。参考不调用新旧自定义算子、ACLNN 或其他后端。
公式等价不代表 Kernel 的中间 BF16 存储、舍入位置或两代 NPU 数值结果已经验证。

## 支持范围与验证状态

| 实现 | 声明与实现范围 | 验证边界 |
| --- | --- | --- |
| `auto`、`ascendc` | arch22（`2201`）、arch35（`3510`）；arch22 为 BF16 状态，arch35 为 BF16/FP32 状态 | 完整 SDK 构建、可选 chunk_state 执行、workspace 兼容性和 NPU 精度仍待验证 |
| `torch_native` | 独立 PyTorch 计算，采用上述 shape/dtype | 参考实现；NPU 精度对照仍待验证 |

正常包导入和公开注册表需要已安装的 NPU 运行环境。公开分发不提供 CPU 回退。

精度验收需要旧版对新版直接比较，并各自对独立参考比较，包括调用方可见的
`chunk_state`。统一调用公共 `assert_accuracy`，不覆盖标准：匹配的 BF16 输出
atol/rtol 为 `0.05`，匹配的 FP32 输出为 `1e-5`，不改变输出 dtype 来选择阈值。
ARC-013 保留 `B=1, T=2048, Nk=8, Nv=16, Dk=Dv=128`、BF16 状态和显式
`128**-0.5`。默认/零 scale、可选 chunk 缓冲布局分别标明为契约变体。
源码实现本身不构成 NPU 精度或性能通过的证据。
