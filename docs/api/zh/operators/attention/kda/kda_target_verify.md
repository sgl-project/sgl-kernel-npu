# kda_target_verify

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention.kda import kda_target_verify


def kda_target_verify(
    *,
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    scale=None,
    gates_are_preactivated=None,
    impl="auto"
): ...
```

## 功能描述

执行固定步数 KDA 投机验证：归一化 Q/K，逐通道衰减递归状态，应用门控 delta 更新，再用 Q 投影更新后的状态。初始状态只读；逐步快照写入指定暂存槽，负源索引表示零初态，负目标索引表示不写快照。支持原始或已激活的门控及分组 Q/K 头。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `A_log` | Tensor | `Required` | 浮点衰减权重 [heads] 或等价单维扩展形状。 |
| `dt_bias` | Tensor | `Required` | GDN 为 [heads]，KDA 为 [Hk,K] 门控偏置。 |
| `q` | Tensor | `Required` | KDA 为 [1,tokens,Hq,K]；FIA 为 FP16/BF16 [tokens,QH,D]。 |
| `k` | Tensor | `Required` | 浮点 key [1,tokens,Hk,K]。 |
| `v` | Tensor | `Required` | 浮点 value [1,tokens,Hv,V]，Hv 须能被 Hq、Hk 整除。 |
| `a` | Tensor | `Required` | GDN 为 [tokens,heads]，KDA 为 [tokens,Hk,K]，可带前导单维。 |
| `b` | Tensor | `Required` | 浮点 [tokens,Hv]；KDA 可为 [1,tokens,Hv]。 |
| `initial_state_source` | Tensor | `Required` | 只读浮点初态 [pool,Hv,V,K]，支持带步幅的 V/K。 |
| `initial_state_indices` | Tensor | `Required` | INT32/INT64 源槽 [batch]，负值表示零初态。 |
| `intermediate_states_buffer` | Tensor | `Required` | 可写 FP32/BF16 快照 [scratch,cache_steps,Hv,V,K]，不得与初态重叠。 |
| `intermediate_state_indices` | Tensor | `Required` | INT32/INT64 目标槽 [batch]，负值跳过写入。 |
| `cache_steps` | int | `Required` | 每个请求的正步数，token 总数须能被其整除。 |
| `scale` | float or None | `None` | QK 缩放；KDA 默认 key_dim**-0.5，要求为正。 |
| `gates_are_preactivated` | bool or None | `None` | true 表示 a 为对数衰减、b 为 sigmoid 输出；None 在二者均有前导单维时推断为 true。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | [1,tokens,Hv,V]，与 v 同类型、同设备；更新快照缓冲区。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.kda import kda_target_verify

q = torch.randn(1, 4, 2, 128, device="npu", dtype=torch.bfloat16)
k, v = torch.randn_like(q), torch.randn_like(q)
a = -torch.rand(1, 4, 2, 128, device="npu")
b = torch.rand(1, 4, 2, device="npu")
initial = torch.zeros(1, 2, 128, 128, device="npu")
snapshots = torch.empty(1, 4, 2, 128, 128, device="npu")
slot = torch.tensor([0], device="npu", dtype=torch.int32)
y = kda_target_verify(
    A_log=torch.zeros(2, device="npu"),
    dt_bias=torch.zeros(2, 128, device="npu"),
    q=q,
    k=k,
    v=v,
    a=a,
    b=b,
    initial_state_source=initial,
    initial_state_indices=slot,
    intermediate_states_buffer=snapshots,
    intermediate_state_indices=slot,
    cache_steps=4,
    gates_are_preactivated=True,
    impl="triton",
)
```
