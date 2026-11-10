# mega_chunk_gdn

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

省略 `impl` 或传入 `"auto"` 时选择 `"ascendc"`，不会自动回退到其他后端。
ARCH35 上可显式指定 `"torch_native"` 进行参考计算，不支持原生 mega 实现。

```python
from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

def mega_chunk_gdn(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, *, impl="auto",
):
    ...
```

## 功能说明

以 128 个 token 分块、head dimension 为 128，计算 gated delta rule 的前向结果。
原生 PTO kernel 融合分块累加、scaled KKT、三角求逆、WY 重算、状态更新和输出计算，
返回注意力输出、最终状态及中间结果。

每 `H/Hg` 个 value head 共享一个 query/key head，packed 输入中的各序列独立维护状态。
`g` 应为非正的对数衰减，`beta` 为 `(0,1]` 范围内的更新权重。本接口不对 Q/K 归一化。

Q/K/V、beta 和初始状态在原生计算前转为 FP16。结果按下表恢复 dtype；
最终状态经过 FP16 舍入后以 FP32 返回。`scale=None` 使用 `128**-0.5`，
显式传入 0 时保留零缩放。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- | --- |
| `q`, `k` | `torch.Tensor` | `[1,T,Hg,128]` | `torch.float16`、`torch.bfloat16` | 必填 | Query 和 key，转为 FP16 计算。 |
| `v` | `torch.Tensor` | `[1,T,H,128]` | `torch.float16`、`torch.bfloat16` | 必填 | Value；`H` 属于 8、12、16、24、32、48、64，且可被 `Hg` 整除。 |
| `g` | `torch.Tensor` | `[1,T,H]` | `torch.float32` | 必填 | 逐 token 的对数衰减，尚未累加。 |
| `beta` | `torch.Tensor` | `[1,T,H]` | `torch.float16`、`torch.bfloat16`、`torch.float32` | 必填 | 状态更新权重，转为 FP16 计算。 |
| `scale` | `float` 或 `None` | — | — | `None` | 输出缩放系数；`None` 对应 `128**-0.5`。 |
| `initial_state` | `torch.Tensor` 或 `None` | `[N,H,128,128]` | `torch.float16`、`torch.bfloat16`、`torch.float32` | `None` | 初始状态，转为 FP16；`None` 表示从零状态开始。 |
| `output_final_state` | `bool` | — | — | `True` | 是否返回最终状态。 |
| `cu_seqlens` | `torch.Tensor` 或 `None` | `[N+1]` | `torch.int32`、`torch.int64` | `None` | Packed 序列边界，转为 INT32；首项为 0、末项为 `T`，序列长度为正。`None` 表示单条序列。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"ascendc"`、`"torch_native"`。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `g_sum` | `torch.Tensor` | `[1,T,H]` | `torch.float32` | 各分块内局部累加的对数衰减。 |
| `o` | `torch.Tensor` | `[1,T,H,128]` | `q.dtype` | 注意力输出；先在 FP16 下乘 scale，再恢复 dtype。 |
| `A_inv` | `torch.Tensor` | `[1,T,H,128]` | `k.dtype` | 按 token、head 存储的分块逆矩阵。 |
| `final_state` | `torch.Tensor` 或 `None` | `[N,H,128,128]` | `torch.float32` | 经过 FP16 舍入的最终状态；`output_final_state=False` 时返回 `None`。 |
| `w` | `torch.Tensor` | `[1,T,H,128]` | `k.dtype` | WY key 变换。 |
| `h` | `torch.Tensor` | `[1,NT,H,128,128]` | `k.dtype` | 进入各分块前的状态，按序列、分块顺序排列。 |
| `v_new` | `torch.Tensor` | `[1,T,H,128]` | `v.dtype` | 减去状态投影后的残差 value。 |

`N` 为序列数，未传入 `cu_seqlens` 时为 1。`NT` 为各序列
`ceil(sequence_length / 128)` 之和。结果位于 `q.device`，输入不被修改。
原生输入必须连续且位于同一 NPU。这些要求由调用方保证，接口仅对其中部分约束进行显式校验。

## ascendc 后端

- 在 ARCH22 上通过 CANN/ACLNN 算子执行融合 PTO 计算。
- 要求安装的 wheel 包含原生 `mega_chunk_gdn` 算子。
  从源码构建时，需要在编译前初始化已锁定版本的 `third_party/pto-isa` 子模块。
- 保留 FP16 计算、中间舍入和指数截断。不会自动将输入转为连续张量；
  直接调用本接口前，应选择输入所在的 NPU。

## torch_native 后端

- 应用相同的入口 FP16 转换后，在 CPU 上使用 FP64 独立分块代数计算，再恢复结果的 dtype 和设备。
- 为两种芯片架构提供精度参考；即使输入、输出位于 NPU，参考计算仍在 CPU 上执行。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

q = torch.randn(1, 129, 4, 128, device="npu", dtype=torch.float16)
k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
v = torch.randn(1, 129, 8, 128, device="npu", dtype=q.dtype)
g = -torch.rand(1, 129, 8, device="npu", dtype=torch.float32)
beta = torch.rand(1, 129, 8, device="npu", dtype=q.dtype) * 0.5 + 0.5
cu_seqlens = torch.tensor([0, 22, 129], device="npu", dtype=torch.int32)
g_sum, o, A_inv, final_state, w, h, v_new = mega_chunk_gdn(
    q, k, v, g, beta, cu_seqlens=cu_seqlens,
)
```
