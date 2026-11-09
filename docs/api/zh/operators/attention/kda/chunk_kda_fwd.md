# chunk_kda_fwd

## 芯片支持

| 实现方式 | ARCH22 | ARCH35 |
| --- | --- | --- |
| ascendc | 支持 | 支持 |
| torch_native | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

def chunk_kda_fwd(
    q, k, v, g, beta, *,
    a_log=None, dt_bias=None, initial_state=None,
    cu_seqlens=None, chunk_indices=None,
    layout="BSND", scale=1.0, chunk_size=64,
    safe_gate=False, lower_bound=-5.0, use_gate_in_kernel=False,
    state_v_first=False, output_final_state=True,
    output_gk=False, output_w=False, output_u=False,
    output_qg=False, output_kg=False, output_v_new=False, output_h=False,
    impl="auto",
):
    ...
```

## 功能描述

执行带逐通道衰减门控的 Kimi Delta Attention（KDA）融合 prefill：计算 chunk 内累计门控和因果系数，求解 delta-rule 变换，跨 chunk 更新循环状态，再重建注意力输出。支持共享 Q/K head、packed 变长序列、64/128 两种分块、可选原始门控激活及中间结果。输入不被原地修改，固定返回 11 项，未请求的中间结果位置保留为 None。

## 参数

| 名称 | 类型 | 形状 / 取值 | 说明 |
| --- | --- | --- | --- |
| q / k | BF16 | BSND: [B,T,H,K]; BNSD: [B,H,T,K]; TND: [T,H,K]; NTD: [H,T,K] | 两者形状一致。H > 0，HV 是 H 的整数倍。 |
| v | BF16 | 相同布局，head 数为 HV，最后一维为 V | 注意力的 value 输入。 |
| g | FP32 / BF16 | 与 v 相同布局，最后一维为 K | 逐通道对数衰减；use_gate_in_kernel=True 时传入未激活的门控。 |
| beta | FP32 / BF16 | v 去掉最后一维 | 已激活的 delta 系数；算子不对 beta 执行 sigmoid。 |
| a_log | FP32 / None | [HV] | 内部激活门控时必须提供。 |
| dt_bias | FP32 / None | [HV*K] | 可选的原始门控偏置。 |
| initial_state | FP32 / None | [N,HV,K,V]；state_v_first=True 时为 [N,HV,V,K] | 每个序列一个初始状态；None 表示从零状态开始。 |
| cu_seqlens | int64 / None | [N+1] | packed 序列边界，从 0 开始、T 结束，非递减，N <= 1024；packed 输入使用 B=1。 |
| chunk_indices | int64 / None | [total_chunks,2] 或展平视图 | 按序列和 chunk 顺序排列的 (序列编号, 序列内 chunk 编号)。需要 cu_seqlens；None 时自动推导。 |
| layout | str | BSND / BNSD / TND / NTD | 默认 BSND；三维布局使用一条 packed token 轴。 |
| scale | float | 标量 | 默认 1.0，通常取 K**-0.5。 |
| chunk_size | int | 64 或 128 | 默认 64。K、V 为 [16,256] 内的 16 的倍数；H、HV 不超过 128；T 为正数。 |
| safe_gate | bool | 标量 | 默认 False。内部激活时，True 使用 lower_bound * sigmoid(exp(a_log) * (g + dt_bias))，False 使用 -exp(a_log) * softplus(g + dt_bias)。 |
| lower_bound | float | 标量 | 默认 -5.0；safe_gate 和 use_gate_in_kernel 同时为 True 时必须位于 [-5,0)。 |
| use_gate_in_kernel | bool | 标量 | 默认 False；为 True 时在内部激活原始 g，不改变 beta。 |
| state_v_first | bool | 标量 | 默认 False；为 True 时，初始、最终和 chunk 状态的最后两维使用 [V,K]。 |
| output_final_state | bool | 标量 | 默认 True；在返回元组的位置 1 返回最终状态。 |
| output_gk / output_w / output_u / output_qg / output_kg / output_v_new / output_h | bool | 标量 | 默认均为 False；未请求的输出位置为 None，元组长度不变。 |
| impl | str | auto / ascendc / torch_native | 仅限关键字传入；默认 auto 选择 ascendc。 |

## 返回值

| 位置 | 名称 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | attn_out | BF16 | [B,T,HV,V]；三维输入时为 [T,HV,V] | 始终返回；即使输入为 BNSD/NTD，输出也采用序列优先顺序。 |
| 1 | final_state | FP32 | [N,HV,K,V] 或 [N,HV,V,K] | output_final_state=False 时为 None。 |
| 2 | gk | FP32 | [B,HV,T,K] 或 [HV,T,K] | chunk 内累计的 log2 衰减；output_gk=False 时为 None。 |
| 3 / 4 | aqk_out / akk_out | BF16 | [B,HV,T,C] 或 [HV,T,C] | 缩放后的因果 Q/K 分数，以及求解后的 K/K 下三角系数，始终返回。 |
| 5 / 7 / 8 | w / qg / kg | BF16 | [B,HV,T,K] 或 [HV,T,K] | 分别由对应的输出开关控制。 |
| 6 / 9 | u / v_new | BF16 | [B,HV,T,V] 或 [HV,T,V] | 分别由对应的输出开关控制。 |
| 10 | h | BF16 | [B,NC,HV,K,V] 或 [NC,HV,K,V] | 每个 chunk 进入时的状态；state_v_first=True 时交换最后 K/V 两维。packed 输入的 NC 为 total_chunks；output_h=False 时为 None。 |

## ascendc

支持 ARCH22 和 ARCH35。Q/K/V 使用 BF16，g 和 beta 分别支持 FP32 或 BF16；状态和中间输出布局见参数及返回值表。

## torch_native

提供相同参数与输出开关的 PyTorch 实现，使用 FP32 计算并保留 BF16 中间结果及输出边界，输出位于输入设备。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

q = torch.nn.functional.normalize(torch.randn(1, 64, 1, 128, device="npu"), dim=-1).bfloat16()
k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
v = torch.randn_like(q)
g = torch.full(q.shape, -0.01, device=q.device, dtype=torch.float32)
beta = torch.full(q.shape[:-1], 0.5, device=q.device, dtype=torch.float32)
outputs = chunk_kda_fwd(q, k, v, g, beta, scale=128**-0.5, impl="ascendc")
out, final_state = outputs[:2]
print(out.shape, final_state.shape, len(outputs))
# torch.Size([1, 64, 1, 128]) torch.Size([1, 1, 128, 128]) 11
```
