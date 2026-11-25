# lp_post

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.moe import lp_post


def lp_post(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
    *,
    impl="auto"
): ...
```

## 功能描述

把 LPLB 内点法解转换为各逻辑专家的副本概率表。先由两个互不重叠的来源组装每个物理槽位的概率，再通过逻辑到物理的映射表取回输出：

```text
phy_prob[:]                  = 0
phy_prob[phy_replicated]     = max(x, 0)
phy_prob[phy_single]         = t1
log2phy_prob[i, j]           = phy_prob[log2phy[i, j] < 0 ? P - 1 : log2phy[i, j]]
```

其中 `P = num_single + num_red_phy + 1`，多出的一个槽位是恒为零的 sink 槽，因此 `log2phy` 中 `-1` 填充项无需分支即可读到 0。经过本算子后，`log2phy_prob` 每一行的和等于该逻辑专家在 LP 求解后仍可复制的流量比例。

## 参数与返回值

所有张量须位于同一 NPU 设备且连续。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `log2phy_prob` | Tensor | `Required` | 可写 float32 [num_logical,max_copies] 输出概率表。 |
| `x` | Tensor | `Required` | float32 [num_red_phy] 副本比例，通常取 `solve_ipm` 结果的 `solution[:num_red_phy]`；负值截断为 0。 |
| `t1` | Tensor | `Required` | float32 [num_single] 由 `lp_prep` 产生的单副本份额。 |
| `phy_single` | Tensor | `Required` | int32/int64 [num_single] 各单副本专家的物理槽位。 |
| `phy_replicated` | Tensor | `Required` | int32/int64 [num_red_phy] 各副本的物理槽位。 |
| `log2phy` | Tensor | `Required` | int32/int64 [num_logical,max_copies] 逻辑到物理映射表；`-1` 表示未使用的副本。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

`phy_single` 与 `phy_replicated` 必须互不重叠，且每个元素都在 `[0, P - 1]` 内。

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| log2phy_prob | Tensor | 同一缓冲区，已被副本概率覆盖。 |

## triton 后端

- 两次下发。第一次用“比较求和”的逆散射方式构建物理概率表：每个槽位累加所有指向它的来源通道。这样避开了 Ascend 上逐 lane 的标量写路径；由于两个来源互不重叠，结果与顺序 scatter 逐位一致。
- 零初始化通过覆盖完整 `[0, P)` 区间融合进第一次下发，不需要单独的 memset 内核。
- 第二次下发将整张表一次性载入 UB，用硬件 gather 路径完成 `log2phy_prob` 的取值；`-1` 项回绕到 sink 槽位。

## torch_native 后端

- 用 `zeros` 加两次按索引赋值构建 `phy_prob`，用 `clamp(min=0)` 截断 `x`，把 `-1` 替换为 sink 槽位后用 `phy_prob[log2phy]` 取出输出表。
- 与内核实现相互独立，可作为精度基准。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import lp_post

num_logical, max_copies = 3, 2
log2phy = torch.tensor([[0, 1], [2, -1], [3, -1]], dtype=torch.int32, device="npu")
log2phy_prob = torch.zeros(num_logical, max_copies, device="npu")
x = torch.tensor([0.4, 0.6], device="npu")  # solve_ipm 得到的副本比例
t1 = torch.tensor([0.2], device="npu")  # lp_prep 得到的单副本份额
phy_single = torch.tensor([3], dtype=torch.int32, device="npu")
phy_replicated = torch.tensor([0, 2], dtype=torch.int32, device="npu")

lp_post(log2phy_prob, x, t1, phy_single, phy_replicated, log2phy, impl="triton")
```
