# mega_chunk_gdn

## 硬件支持

| 实现 | ARCH22 | ARCH35 |
| --- | --- | --- |
| `ascendc`（也是 `auto`） | 已声明；CANN 编译和设备验证待完成 | 不支持 |
| `torch_native` | 参考实现；设备验证待完成 | 参考实现；设备验证待完成 |

## 函数签名

```python
from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

def mega_chunk_gdn(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, *, impl="auto",
):
    ...
```

`auto` 选择 `ascendc`，不自动回退。原有 PTO 计算通过独立 CANN/ACLNN 构建接入，
不引入外部 Python 依赖。本次不扩展 ARCH35 支持。

## 功能描述

融合分块累加、scaled KKT、三角求逆、WY 重算、状态更新和输出计算。
原 PTO 阶段源码和 Python helper 原样复制。分块大小为 128，head dimension 为 128；
每 H/Hg 个 value head 共享一个 Q/K head。

数学递推为
`S_t = exp(g_t) * S_(t-1) + beta_t * k_t ⊗ (v_t - k_t^T exp(g_t) S_(t-1))`，
`o_t = scale * q_t^T S_t`。提供非正对数衰减以及 (0,1] 范围内的 beta。
原生实现保留既有 FP16 舍入和指数截断，不新增 Q/K 归一化。

## 参数与返回值

| 参数 | 类型 / 形状 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `q`, `k` | Tensor `[1,T,Hg,128]` | 必填 | contiguous FP16/BF16 NPU 输入，转为 FP16。 |
| `v` | Tensor `[1,T,H,128]` | 必填 | 同设备，转为 FP16；H 属于 {8,12,16,24,32,48,64} 且可被 Hg 整除。 |
| `g` | FP32 Tensor `[1,T,H]` | 必填 | contiguous 逐 token 对数衰减。 |
| `beta` | Tensor `[1,T,H]` | 必填 | contiguous 更新权重，转为 FP16。 |
| `scale` | float 或 None | None | None 对应 `128**-0.5`，保留显式零值。 |
| `initial_state` | Tensor `[N,H,128,128]` 或 None | None | 转为 FP16；None 对应零初始状态。 |
| `output_final_state` | bool | True | 是否返回最终状态，否则返回 None。 |
| `cu_seqlens` | Tensor `[N+1]` 或 None | None | contiguous 边界转为 INT32；首项 0、末项 T，长度为正。None 对应 [0,T]。 |
| `impl` | str | `"auto"` | `auto`、`ascendc` 或 `torch_native`。 |

所有 Tensor 必须在同一 NPU 上且 contiguous；本 helper 不增加 input guard。
输入不被修改。FP16 原生结果在与旧 helper 相同的位置恢复原 dtype。

| 返回值 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `g_sum` | `[1,T,H]`，FP32 | 每个分块内局部累加的对数衰减。 |
| `o` | `[1,T,H,128]`，q.dtype | 输出，先在 FP16 下乘 scale 再恢复 dtype。 |
| `A_inv` | `[1,T,H,128]`，k.dtype | 按 token/head 存储的分块逆矩阵。 |
| `final_state` | `[N,H,128,128]`，FP32 或 None | 经原生 FP16 舍入后的最终状态。 |
| `w` | `[1,T,H,128]`，k.dtype | WY key 变换。 |
| `h` | `[1,total_chunks,H,128,128]`，k.dtype | 按序列、分块排列的进入分块前的状态。 |
| `v_new` | `[1,T,H,128]`，v.dtype | 减去状态投影后的残差 value。 |

## Ascend C 实现

通过独立原生构建选择本仓库算子，不依赖 legacy 构建：

```bash
git submodule update --init third_party/pto-isa
python3 setup.py --arch-version 2201 --ops mega_chunk_gdn --without-third-party
```

使用仓库已锁定的 PTO-ISA 头文件。参考已有 attention 域算子，分离 OpDef、推导、Tiling、
Kernel 入口和 Torch adapter。原 scalar launch 参数通过 Tiling 传入，保留 block 数和 PTO 分发。
19 个可写 buffer 使用同名 CANN Ref 端口，保留 helper 的初始零值和 scratch buffer。
原计算包含跨核同步；完整 SDK 编译、launch、stream/graph 行为及设备精度仍需 ARCH22 验证。
独立 adapter 保留 `torch.ops.npu.mega_chunk_gdn` schema；legacy 与独立构建的 wheel 分别安装使用。

## PyTorch 参考实现

`torch_native` 先应用相同的入口 FP16 转换，再用 CPU FP64 独立分块代数计算，并恢复结果的
设备和 dtype。它用于精度比较，不用于性能测试或逐指令模拟 PTO。测试沿用旧 mega 测试的
最大绝对误差或相对 RMS 阈值；参考实现还与独立逐 token 递推进行了核对。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

q = torch.randn(1, 129, 4, 128, device="npu", dtype=torch.float16)
k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
v = torch.randn(1, 129, 8, 128, device="npu", dtype=q.dtype)
g = -torch.rand(1, 129, 8, device="npu", dtype=torch.float32)
beta = torch.rand(1, 129, 8, device="npu", dtype=q.dtype) * 0.5 + 0.5
cu = torch.tensor([0, 22, 129], device="npu", dtype=torch.int32)
g_sum, o, A, final_state, w, h, v_new = mega_chunk_gdn(
    q, k, v, g, beta, cu_seqlens=cu, impl="ascendc",
)
```
