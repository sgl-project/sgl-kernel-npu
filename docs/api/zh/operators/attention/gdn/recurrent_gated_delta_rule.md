# recurrent_gated_delta_rule

## 芯片支持

| 实现 | arch 2201 | arch 3510 |
| --- | --- | --- |
| `ascendc` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认 `auto` 与 `ascendc` 使用同一入口。

```python
from sgl_kernel_npu.operators.attention.gdn import recurrent_gated_delta_rule

def recurrent_gated_delta_rule(
    mix_qkv, recurrent_state, beta, scale, actual_seq_lengths,
    ssm_state_indices, nk, nv, intermediate_state=None,
    cache_indices=None, num_accepted_tokens=None, g=None, gk=None,
    *, impl="auto",
):
    ...
```

旧接口名 `torch.ops.npu.recurrent_gated_delta_rule` 保留。
新公共接口的 `auto` / `ascendc` 通过该名称调用当前安装包的实现；
`torch_native` 使用独立递推公式。

## 功能说明

value 头 `h` 使用 Q/K 头 `h // (Nv // Nk)`：

```text
q = scale * q_raw / sqrt(sum(q_raw*q_raw) + 1e-6)
k = k_raw / sqrt(sum(k_raw*k_raw) + 1e-6)
D_t = exp(g_t) * H_(t-1)                 # g=None 时 D_t = H_(t-1)
delta_t = sigmoid(beta_t) * (v_t - D_t @ k_t)
H_t = D_t + outer(delta_t, k_t)
out_t = H_t @ q_t
```

状态矩阵布局为 `[Dv,Dk]`。连续 token 之间保留 FP32 状态；每个 token 写出 BF16
快照，下一步不重新读入舍入后的快照。

1. 没有中间池时，读取并回写 `recurrent_state`。
2. 有中间池时，读取并回写中间工作池；两个池不重叠时，原 recurrent 池只提供种子，
   保持不变。
3. 有 accepted 计数时初始槽为 `indices[b,accepted[b]-1]`，否则为 `indices[b,0]`。
   accepted 计数必须在 `[1,S]` 内。
4. 中间池模式下，所选槽能被 `S` 整除时，从 `recurrent_state[cache_indices[b]]`
   初始化；否则从所选中间槽初始化。第 `t` 个有效 token 写入工作池的 `indices[b,t]`。

## 参数与返回值

| 参数 | Shape / Dtype | 说明 |
| --- | --- | --- |
| `mix_qkv` | `[B,S,2*Nk*Dk+Nv*Dv]`，BF16 | 按 Q、K、V 顺序拼接。 |
| `recurrent_state` | `[N,Nv,Dv,Dk]`，BF16 | 普通模式下读写的状态池；提供中间池时作为 recurrent 种子池。 |
| `beta` | `[B,S,Nv]`，BF16 | sigmoid 之前的值。 |
| `scale` | 必填 float | Q 的缩放系数，保留 `0.0`，沿用旧 Host 的 FP32 窄化。 |
| `actual_seq_lengths` | `[B]`，int32 | 每条序列的长度，不是累积边界；序列起点固定为 `b*S`。 |
| `ssm_state_indices` | `[B,S]`，int32 | 初始状态选择及各 token 的回写槽位。 |
| `nk`、`nv` | 正整数 | Q/K 与 value 头数，要求 `Nv % Nk == 0`。 |
| `intermediate_state` | 完整的 `[Nv,Dv,Dk]` 槽位，BF16 / `None` | 默认 `None`；工作池通常为 `[N*S,Nv,Dv,Dk]`。 |
| `cache_indices` | 至少 `B` 个元素 / `None` | 默认 `None`；每条序列的 recurrent 种子槽；提供中间池时必填；转为连续 int32。 |
| `num_accepted_tokens` | 至少 `B` 个元素 / `None` | 默认 `None`（第 0 列）；选择初始列 `accepted[b]-1`，转为连续 int32；不缩短本次计算长度。 |
| `g` | 至少 `B*S*Nv` 个元素 / `None` | 默认 `None`；log decay，转为连续 FP32 后取 exp；缺省时衰减系数为 1。 |
| `gk` | 可选 Tensor | 默认 `None`；为兼容旧接口转为连续 FP32；旧计算逻辑不使用其数值。 |
| `impl` | `str`，默认 `"auto"` | 仅限关键字，可选 `auto`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | Shape | Dtype | 说明 |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[B,S,Nv,Dv]` | BF16 | 新分配输出，仅有效长度以内的 token 有定义；状态池原地更新。 |

维度必须为正，`1 <= S <= 8`，有效长度
在 `[1,S]` 内。`ascendc` 要求所有张量位于同一 NPU；`torch_native` 要求同一设备。
槽位必须在对应池的范围内；不同序列不能发生冲突的状态读写。原生校验只检查元数据，
不把设备张量中的索引、长度值同步回 Host 做检查。

## ascendc 后端

必填的 QKV、beta、lengths、indices、recurrent_state 保留旧 `data_ptr()` 语义：
从 storage offset 开始按物理连续区间读取，不按 Tensor 的逻辑 stride 重排，
也不新增 contiguous 拷贝；底层 storage 必须覆盖该区间。可选索引、gate 按旧行为
转换 dtype 并连续化。连续中间池直接回写；非连续中间池使用临时副本，**不拷回原池**，
与旧 Host 一致。需要观察状态更新时应传连续状态池。
原生 beta 拷贝要求 `S*Nv*2 <= 65535` 字节；提供 `g` 时还要求
`S*Nv*4 <= 65535` 字节。arch 2201 对齐后的 FP32 行步长不能超过 255 个
32 字节块。不满足时在启动 kernel 前报错。

## torch_native 后端

使用独立 PyTorch 递推，以 FP32 保留中间状态，输出和状态快照为 BF16。
遵从相同的状态选择与物理区间语义；参考实现不受原生拷贝字段大小限制。

## 调用示例

沿用现有 `(B,S,Nk,Nv,Dk,Dv)=(2,1,8,16,128,128)` 配置，
两后端各自使用独立的初始状态池。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import recurrent_gated_delta_rule

b, s, nk, nv, dk, dv = 2, 1, 8, 16, 128, 128
device, dtype = "npu", torch.bfloat16
q = torch.randn(b, s, nk, dk, device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn(b, s, nv, dv, device=device, dtype=dtype)
state = 0.1 * torch.randn(b * s, nv, dv, dk, device=device, dtype=dtype)
state_ref = state.clone()
inputs = dict(
    mix_qkv=torch.cat([q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1),
    beta=torch.randn(b, s, nv, device=device, dtype=dtype),
    scale=dk**-0.5,
    actual_seq_lengths=torch.full((b,), s, device=device, dtype=torch.int32),
    ssm_state_indices=torch.arange(b * s, device=device, dtype=torch.int32).view(b, s),
    nk=nk,
    nv=nv,
    g=-(0.05 + 0.95 * torch.rand(b, s, nv, device=device)),
)
out = recurrent_gated_delta_rule(**inputs, recurrent_state=state)  # auto: ascendc
out_ref = recurrent_gated_delta_rule(
    **inputs, recurrent_state=state_ref, impl="torch_native"
)
print(out.shape, out_ref.shape)  # torch.Size([2, 1, 16, 128])
```
