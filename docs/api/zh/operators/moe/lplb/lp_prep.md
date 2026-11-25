# lp_prep

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.moe import lp_prep


def lp_prep(
    A_full,
    b,
    t1,
    global_counts,
    log_single,
    log_replicated,
    B1,
    A_base_row_sum,
    *,
    impl="auto"
): ...
```

## 功能描述

构建 LPLB（负载均衡放置）求解器的内点法输入。先按总量归一化各逻辑专家的 token 计数，拆出单副本份额 `t1` 与多副本右端项 `b1`，其中 GPU 行的右端项为 `b2 = -(B1 @ t1)`；再把两者写入 `A_full` 的最后一列，该列存放标准型 LP 取负后的右端项。

```text
total     = max(sum(global_counts), 1)
t1        = global_counts[log_single] / total
b1        = global_counts[log_replicated] / total
b2        = -(B1 @ t1)
b         = concat(b1, b2)
A_full[:, NV - 1] = b - A_base_row_sum
```

`A_full` 由调用方预分配，其前 `NV - 1` 列已经存放约束矩阵的常量部分，本算子只写最后一列。`b` 与 `t1` 原地覆盖，调用方可以在各步之间复用长生命周期缓冲区。

## 参数与返回值

所有张量须位于同一 NPU 设备且连续。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `A_full` | Tensor | `Required` | 可写 float32 [NC,NV]；只写最后一列。NC 须等于 `len(log_replicated) + B1.shape[0]`。 |
| `b` | Tensor | `Required` | 可写 float32 [NC] 右端项缓冲区。 |
| `t1` | Tensor | `Required` | 可写 float32 [num_single] 单副本份额缓冲区。 |
| `global_counts` | Tensor | `Required` | float32 [num_logical] 每个逻辑专家的 token 计数。 |
| `log_single` | Tensor | `Required` | int32/int64 [num_single] 只有单个物理副本的逻辑专家 id。 |
| `log_replicated` | Tensor | `Required` | int32/int64 [num_red_log] 拥有多个物理副本的逻辑专家 id。 |
| `B1` | Tensor | `Required` | float32 [num_gpus,num_single] 单副本专家的 GPU 分配矩阵。 |
| `A_base_row_sum` | Tensor | `Required` | float32 [NC] 约束矩阵常量部分的行和。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| A_full | Tensor | 同一缓冲区，最后一列已被覆盖。 |
| b | Tensor | 同一缓冲区，已被 `concat(b1, b2)` 覆盖。 |
| t1 | Tensor | 同一缓冲区，已被单副本份额覆盖。 |

## triton 后端

- 单 program 内核（`grid=(1,)`）：计数表一次性载入 UB，各专家的计数通过 UB 内 gather 选取，不产生随机全局读。
- `B1 @ t1` 按 GPU 数量选择的行块一次展开完成，`t1` / `b` 的写入融合在同一次下发中。
- 各 tile 的填充通道对归约的贡献严格为零，因此专家数量不必是 2 的幂。
- 填充后的 `BS`/`BR`/`BT` 必须位于 32768 个常驻元素以内；更大的输入会在启动前以 `ValueError` 拒绝。

## torch_native 后端

- 用张量表达式直接实现上述公式（`sum`、`clamp`、按索引取值、矩阵向量乘），与内核实现相互独立。
- 遵循相同的原地写回约定，因此可作为 triton 路径的精度基准。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import lp_prep

num_gpus, num_single, num_red_log = 2, 2, 1
nc, nv = num_red_log + num_gpus, 2 * num_single + num_gpus + 1

A_full = torch.zeros(nc, nv, dtype=torch.float32, device="npu")
b = torch.empty(nc, dtype=torch.float32, device="npu")
t1 = torch.empty(num_single, dtype=torch.float32, device="npu")
counts = torch.tensor([20.0, 5.0, 7.0], device="npu")
log_single = torch.tensor([1, 2], dtype=torch.int32, device="npu")
log_replicated = torch.tensor([0], dtype=torch.int32, device="npu")
B1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="npu")

lp_prep(
    A_full,
    b,
    t1,
    counts,
    log_single,
    log_replicated,
    B1,
    A_full.sum(1).contiguous(),
    impl="triton",
)
```
