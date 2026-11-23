"""Independent torch reference for the LPLB interior-point solver."""

import torch


def solve_ipm(A, b, c, num_iters=5, result=None):
    """Return the barrier-method solution of the standard-form LP.

    Step-for-step port of ``sglang/kernels/jit/csrc/lplb/ipm.cuh``. The KKT
    system is solved with ``torch.linalg.solve`` (LU) instead of the kernel's
    in-UB elimination sweep, so the two agree to float32 tolerance rather than
    bit-for-bit. A tiny diagonal jitter mirrors the kernel's pivot clamp and
    keeps a (near-)singular system solvable instead of raising.

    Returns the ``[NV]`` float32 solution; every entry is 0.5 when the
    convergence test fails.
    """
    nc, nv = A.shape
    eye = torch.eye(nc, dtype=torch.float32, device=A.device)
    x = torch.ones(nv, dtype=torch.float32, device=A.device)
    d_max = torch.zeros((), dtype=torch.float32, device=A.device)
    for _ in range(num_iters):
        ax2 = A * (x * x).unsqueeze(0)  # (NC, NV)
        ax2c = ax2 @ c  # (NC,)
        ax2a = ax2 @ A.t() + 1e-12 * eye  # (NC, NC)
        delta = torch.linalg.solve(ax2a, ax2c)  # (NC,)
        r = A.t() @ delta  # (NV,)
        d = x * (c - r)  # (NV,)
        d_max = d.max()
        alpha = (
            0.999 / d_max
            if bool(d_max > 1e-9)
            else torch.ones((), dtype=torch.float32, device=A.device)
        )
        x = x * (1.0 - alpha * d)

    residual = (A @ x - b).abs().max()
    converged = (
        bool(d_max < 0.1)
        and bool(x[-1] >= 0)
        and bool(x[-1] < 1e-4)
        and bool(residual < 0.05)
    )
    solution = (
        x if converged else torch.full((nv,), 0.5, dtype=torch.float32, device=A.device)
    )
    if result is not None:
        result.copy_(solution)
        return result
    return solution
