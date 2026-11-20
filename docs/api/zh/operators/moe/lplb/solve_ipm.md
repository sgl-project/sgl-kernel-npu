# solve_ipm

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `triton_cann920` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

`triton_cann920` 要求 CANN 9.2.x；在其他版本上显式选择会在启动前抛出 `RuntimeError`。

## 函数签名

`auto` 在 CANN 9.2.x 上选择 `triton_cann920`，其他版本选择 `triton`。

```python
from sgl_kernel_npu.operators.moe import solve_ipm


def solve_ipm(A, b, c, num_iters=5, result=None, *, impl="auto"): ...
```

## 功能描述

用原始障碍法固定迭代次数求解 LPLB 放置步骤的标准型线性规划：

```text
min c^T x   subject to   A x = b,  x >= 0

x <- 1
for _ in range(num_iters):
    ax2   = A * x^2                 # (NC, NV)
    ax2a  = ax2 @ A^T               # (NC, NC) KKT 矩阵
    delta = solve(ax2a, ax2 @ c)
    r     = delta^T @ A             # (NV,)
    d     = x * (c - r)
    alpha = 0.999 / max(d)          # max(d) <= 1e-9 时取 1.0
    x     = x * (1 - alpha * d)
```

只有当最后一次迭代后满足 `max(d) < 0.1`、`x[-1]` 位于 `[0, 1e-4)` 且 `max|A x - b| < 0.05` 时才接受该解；否则输出全部置为 `0.5`，作为“LP 未解出、本次不做重平衡”的信号返回给调用方。

Ascend 向量单元没有分块 Cholesky 原语，因此 KKT 系统用 UB 内的对称 `LDL^T` 消元扫描分解。该扫描与基于 LU 的参考实现按 float32 精度一致，而非逐位一致；固定的 `1e-12` 主元下限也会在障碍法把某个变量压到零时限制逆的增长，因此极度收敛的问题可能与双精度求解结果不同。

## 参数与返回值

所有张量须位于同一 NPU 设备且连续。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `A` | Tensor | `Required` | float32 [NC,NV] 约束矩阵，由 `lp_prep` 构建。 |
| `b` | Tensor | `Required` | float32 [NC] 右端项，由 `lp_prep` 构建。 |
| `c` | Tensor | `Required` | float32 [NV] 目标系数；最后一项是消去 sink 变量的大惩罚。 |
| `num_iters` | int | `5` | 障碍法迭代次数；取 `0` 时直接返回全 `0.5` 哨兵值。 |
| `result` | Tensor \| None | `None` | 可选的可写 float32 [NV] 输出缓冲区；省略时新分配。 |
| `impl` | str | `'auto'` | auto 在 CANN 9.2.x 上选择 triton_cann920，其他版本选择 triton；也可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| x | Tensor | float32 [NV] 解；收敛判定失败时全为 `0.5`。 |

## triton 后端

- 整个求解只用一个 program（`grid=(1,)`）：`NC x NV` 矩阵及其转置一次性载入 UB，并在迭代之间常驻。
- 当 tile 足够大时 KKT 乘用 `tl.dot`，否则回退为显式广播归约；填充的行列贡献为零。
- 消元扫描把 `L` 保存在工作 tile 的下三角，通过掩码 rank-1 更新完成 Schur 补，随后做逐行的 `L^T` 回代。
- 填充后的 `A`/`A^T` 与 KKT tile 在启动前按 32768 元素的常驻预算校验；广播回退路径另外限制为 16384 元素。

## triton_cann920 后端

- 与 `triton` 算法和契约相同，但前向、后向消元扫描用 `tl.static_range` 完全展开，并缓存 launcher；这在 CANN 9.2.x 上编译开销很低。
- `auto` 在 CANN 9.2.x 上选择此后端；在其他版本上显式选择会在启动前被拒绝。
- 启动前同样执行常驻 tile 与广播 tile 的预算校验。

## torch_native 后端

- 逐步骤移植 GPU 参考实现，KKT 系统用 `torch.linalg.solve` 求解，保留相同的 `1e-12` 对角抖动与收敛判定。
- `torch.linalg.solve` 在 NPU 上可能回退到 CPU 执行；参考实现仍然确定且与内核无关。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import solve_ipm

# A * x = b，其中 x = [0.25, 0.75]，并带一个惩罚列。
A = torch.tensor([[1.0, 1.0, -1.0], [1.0, 0.0, 0.0]], device="npu")
b = torch.tensor([1.0, 0.25], device="npu")
c = torch.tensor([0.0, 0.0, 1000.0], device="npu")

x = solve_ipm(A, b, c, num_iters=5, impl="triton")
print(x)
```

生产环境中的 `A`、`b`、`c` 布局来自 `lp_prep`，其 `A_full` 的最后一列已经存放取负后的右端项。
