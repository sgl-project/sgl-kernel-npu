# solve_ipm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `triton_cann920` | Supported | Supported |
| `torch_native` | Supported | Supported |

`triton_cann920` requires CANN 9.2.x; selecting it explicitly on another release raises `RuntimeError`.

## Signature

`auto` selects `triton_cann920` on CANN 9.2.x and `triton` on other releases.

```python
from sgl_kernel_npu.operators.moe import solve_ipm


def solve_ipm(A, b, c, num_iters=5, result=None, *, impl="auto"): ...
```

## Description

Solves the standard-form linear program of the LPLB placement step with a primal barrier method for a fixed number of iterations:

```text
min c^T x   subject to   A x = b,  x >= 0

x <- 1
for _ in range(num_iters):
    ax2   = A * x^2                 # (NC, NV)
    ax2a  = ax2 @ A^T               # (NC, NC) KKT matrix
    delta = solve(ax2a, ax2 @ c)
    r     = delta^T @ A             # (NV,)
    d     = x * (c - r)
    alpha = 0.999 / max(d)          # 1.0 when max(d) <= 1e-9
    x     = x * (1 - alpha * d)
```

The result is accepted only when `max(d) < 0.1`, `x[-1] in [0, 1e-4)` and `max|A x - b| < 0.05` after the last iteration; otherwise every entry of the output is set to `0.5`, which is the caller's signal that the LP was not solved and no rebalancing should be applied.

The KKT system is factored with an in-UB symmetric `LDL^T` elimination sweep because the Ascend vector unit has no block Cholesky primitive. The sweep therefore agrees with the LU-based reference to float32 tolerance rather than bit-for-bit; the fixed `1e-12` pivot floor also bounds the inverse when the barrier drives a variable to zero, so extremely converged problems can differ from a double-precision solve.

## Parameters and return value

All tensors must be on the same NPU device and contiguous.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `A` | Tensor | `Required` | Float32 [NC,NV] constraint matrix built by `lp_prep`. |
| `b` | Tensor | `Required` | Float32 [NC] right-hand side built by `lp_prep`. |
| `c` | Tensor | `Required` | Float32 [NV] objective coefficients; the last entry is the large penalty that removes the sink variable. |
| `num_iters` | int | `5` | Number of barrier iterations; `0` returns the all-`0.5` sentinel. |
| `result` | Tensor \| None | `None` | Optional writable float32 [NV] output buffer. A fresh tensor is allocated when omitted. |
| `impl` | str | `'auto'` | auto selects triton_cann920 on CANN 9.2.x and triton otherwise; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| x | Tensor | Float32 [NV] solution, or all `0.5` when the convergence test fails. |

## triton backend

- One program for the whole solve (`grid=(1,)`); the `NC x NV` matrix and its transpose are loaded into UB once and stay resident across iterations.
- The KKT product uses `tl.dot` when the tiles are large enough and fall back to an explicit broadcast reduction otherwise; padded rows and columns contribute zero.
- The elimination sweep keeps `L` in the lower triangle of the working tile and applies the Schur complement through masked rank-1 updates, followed by a row-wise `L^T` back substitution.
- Padded `A`/`A^T` and KKT tiles are checked against a 32768-element resident budget before launch; the broadcast fallback is additionally limited to 16384 elements.

## triton_cann920 backend

- Same algorithm and contract as `triton`, but the forward and backward elimination sweeps are fully unrolled with `tl.static_range` and the launcher is cached, which is cheap to compile on CANN 9.2.x.
- `auto` selects this backend on CANN 9.2.x; an explicit request on another release is rejected before launch.
- The same resident and broadcast tile budget checks run before launch.

## torch_native backend

- Step-for-step port of the GPU reference using `torch.linalg.solve` for the KKT system, with the same `1e-12` diagonal jitter and the same convergence test.
- `torch.linalg.solve` may fall back to the CPU on NPU; the reference is still deterministic and independent of the kernel.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import solve_ipm

# A * x = b with x = [0.25, 0.75] and one penalty column.
A = torch.tensor([[1.0, 1.0, -1.0], [1.0, 0.0, 0.0]], device="npu")
b = torch.tensor([1.0, 0.25], device="npu")
c = torch.tensor([0.0, 0.0, 1000.0], device="npu")

x = solve_ipm(A, b, c, num_iters=5, impl="triton")
print(x)
```

The `A`, `b`, and `c` layouts used in production come from `lp_prep`, whose `A_full` already carries the negated right-hand side in its last column.
