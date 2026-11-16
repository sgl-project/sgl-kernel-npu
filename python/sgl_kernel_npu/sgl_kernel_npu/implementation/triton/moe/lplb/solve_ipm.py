"""Triton-Ascend interior-point solver for the LPLB linear program.

Solves ``min c^T x  subject to  A x = b, x >= 0`` with a barrier method for a
fixed number of iterations. The kernel is a single-program port of the GPU
implementation in ``sglang/kernels/jit/csrc/lplb/ipm.cuh``:

    x <- 1
    for _ in range(num_iters):
        ax2   = A * x^2                       # (NC, NV)
        ax2a  = ax2 @ A^T                     # (NC, NC) KKT matrix
        delta = solve(ax2a, ax2 @ c)
        r     = delta^T @ A                   # (NV,)
        d     = x * (c - r)
        alpha = 0.999 / d_max  (or 1.0 when d_max <= 1e-9)
        x    *= 1 - alpha * d
    write 0.5 everywhere on non-convergence.

The KKT system is factored with an in-UB symmetric LDL^T elimination sweep (the
Ascend vector unit has no block Cholesky primitive), so the result matches the
torch reference to float32 tolerance rather than bit-for-bit. Non-convergence is
reported by returning 0.5 for every entry, exactly like the CUDA kernel.

The CANN 9.2 variant (``solve_ipm_cann92``) runs the same algorithm with the
forward and backward sweeps fully unrolled through ``tl.static_range`` (cheap to
compile there, while the 9.1 path needs the rolled ``tl.range`` form).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def ipm_kernel(
    A_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    NUM_ITERS: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    NCP: tl.constexpr,
    NVP: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    oc = tl.arange(0, NCP).to(tl.int32)
    ov = tl.arange(0, NVP).to(tl.int32)

    m_a = (oc[:, None] < NC) & (ov[None, :] < NV)
    m_at = (ov[:, None] < NV) & (oc[None, :] < NC)

    # A (NC, NV) row-major plus its transposed view. ``at_t`` is materialised
    # once so that every A-side contraction can reduce along the contiguous
    # (axis=1) direction instead of a strided axis=0 sweep.
    a_t = tl.load(A_ptr + oc[:, None] * NV + ov[None, :], mask=m_a, other=0.0)
    at_t = tl.load(A_ptr + oc[None, :] * NV + ov[:, None], mask=m_at, other=0.0)
    b_t = tl.load(b_ptr + oc, mask=oc < NC, other=0.0)
    c_t = tl.load(c_ptr + ov, mask=ov < NV, other=0.0)

    eye = tl.where((oc[:, None] == oc[None, :]) & (oc[:, None] < NC), 1.0, 0.0)

    # fp32 lane masks (0/1). Multiplying by these is vector-unit work on
    # Ascend; ``tl.where(<int compare>, <data tile>, 0.0)`` scalarises the whole
    # select, so it is avoided everywhere on the hot path.
    xmask = tl.where(ov < NV, 1.0, 0.0)
    negbig = tl.where(ov < NV, 0.0, -3.0e38)
    lmask = tl.where(ov == NV - 1, 1.0, 0.0)

    x = xmask
    d_max = 0.0

    for _ in range(NUM_ITERS):
        x2 = x * x
        ax2 = a_t * x2[None, :]
        if USE_DOT:
            ax2a = tl.dot(ax2, at_t) + 1e-12 * eye
        else:
            ax2a = tl.sum(ax2[:, None, :] * a_t[None, :, :], axis=2) + 1e-12 * eye
        ax2c = tl.sum(ax2 * c_t[None, :], axis=1)

        # ---- forward: in-place LDL^T elimination of the symmetric KKT matrix ----
        # ``a`` starts as the full symmetric matrix. Step k keeps the trailing
        # Schur complement correct by updating only rows/cols > k with
        #   a[j, l] -= L[j, k] * a[k, l]
        # and then storing L[:, k] in column k (unit diagonal implied, i.e. the
        # diagonal entry and everything above it are zeroed). The backward sweep
        # below reads L back row-wise, so column k must not be touched again by
        # later steps -- ``upper`` guarantees that.
        # ``tl.range`` keeps the sweep a rolled loop: unrolling it with
        # ``tl.static_range`` makes Triton-Ascend compilation of the 48x82
        # production shape take over 15 minutes instead of a few seconds.
        a = ax2a
        dinv = tl.zeros([NCP], dtype=tl.float32) + 1.0
        z = ax2c
        for k in tl.range(0, NC):
            mcol = tl.where(oc[None, :] == k, 1.0, 0.0)  # [1, NCP]
            upper = tl.where(oc[None, :] > k, 1.0, 0.0)  # [1, NCP]
            mrow = tl.where(oc == k, 1.0, 0.0)  # [NCP]
            nrow = tl.where(oc == k, 0.0, 1.0)  # [NCP]
            lower = tl.where(oc > k, 1.0, 0.0)  # [NCP]

            rowk = tl.sum(a * mrow[:, None], axis=0)  # a[k, :]
            dk = tl.sum(rowk * mrow)  # a[k, k]
            inv = 1.0 / tl.maximum(dk, 1e-12)
            col = tl.sum(a * mcol, axis=1)  # a[:, k]
            lk = (col * inv) * lower  # L[:, k], unit diagonal implied

            a = a - lk[:, None] * rowk[None, :] * upper  # Schur complement
            a = (a - a * mcol) + lk[:, None] * mcol  # column k <- L[:, k]

            dinv = dinv * nrow + inv * mrow
            zi = tl.sum(z * mrow)  # z[k]
            z = z - lk * zi

        # ---- backward: L^T delta = D^-1 z ----
        # ``at`` is the transpose of the stored factor, so column i of ``at`` is
        # row i of ``a`` and therefore L[i, :]; masking to j <= i drops the
        # stale upper-triangle entries the forward sweep never rewrote.
        at = tl.trans(a)
        v = z * dinv
        for i in tl.range(NC - 1, -1, -1):
            mcol = tl.where(oc[None, :] == i, 1.0, 0.0)
            mrow = tl.where(oc == i, 1.0, 0.0)
            crow = tl.where(oc <= i, 1.0, 0.0)  # [NCP]
            ri = (tl.sum(at * mcol, axis=1)) * crow  # L[i, :]
            vi = tl.sum(v * mrow)
            v = v - ri * vi
        delta = v

        r = tl.sum(at_t * delta[None, :], axis=1)
        d = x * (c_t - r)
        d_max = tl.max(d * xmask + negbig)
        alpha = tl.where(d_max > 1e-9, 0.999 / d_max, 1.0)
        x = x * (1.0 - alpha * d)

    # ---- convergence test (mirrors the reference exactly) ----
    ax = tl.sum(a_t * x[None, :], axis=1)
    resid = tl.max(tl.abs(ax - b_t))
    x_last = tl.sum(x * lmask)
    ok = (d_max < 0.1) & (x_last >= 0.0) & (x_last < 1e-4) & (resid < 0.05)
    out_v = tl.where(ok, x, 0.5)
    tl.store(out_ptr + ov, out_v, mask=ov < NV)


def _pow2(v, floor=8):
    """Smallest power of two >= v, floored at the measured best small tile."""
    p = floor
    while p < v:
        p *= 2
    return p


def solve_ipm(A, b, c, num_iters=5, result=None):
    """Return the barrier-method solution of the standard-form LP.

    Args:
        A (torch.Tensor): Constraint matrix [NC, NV], float32, contiguous.
        b (torch.Tensor): Right-hand side [NC], float32, contiguous.
        c (torch.Tensor): Objective coefficients [NV], float32, contiguous.
        num_iters (int): Number of barrier iterations; default 5.
        result (torch.Tensor | None): Optional pre-allocated [NV] float32
            output buffer. A fresh tensor is allocated when omitted.

    Returns:
        x (torch.Tensor): Solution vector [NV], float32. Every entry is 0.5
        when the convergence test fails.
    """
    if A.dim() != 2 or A.dtype != torch.float32:
        raise TypeError("A must be a 2-D float32 tensor")
    if b.dim() != 1 or b.dtype != torch.float32:
        raise TypeError("b must be a 1-D float32 tensor")
    if c.dim() != 1 or c.dtype != torch.float32:
        raise TypeError("c must be a 1-D float32 tensor")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous")
    if not b.is_contiguous():
        raise ValueError("b must be contiguous")
    if not c.is_contiguous():
        raise ValueError("c must be contiguous")
    if isinstance(num_iters, bool) or not isinstance(num_iters, int):
        raise TypeError("num_iters must be an integer")
    if num_iters < 0:
        raise ValueError("num_iters must be nonnegative")

    nc, nv = A.shape
    if nc < 1 or nv < 1:
        raise ValueError(f"A must be non-empty, got shape {(nc, nv)}")
    if b.shape != (nc,):
        raise ValueError(f"b must have shape {(nc,)}, got {tuple(b.shape)}")
    if c.shape != (nv,):
        raise ValueError(f"c must have shape {(nv,)}, got {tuple(c.shape)}")

    if result is None:
        result = torch.empty(nv, dtype=torch.float32, device=A.device)
    else:
        if result.shape != (nv,) or result.dtype != torch.float32:
            raise TypeError(f"result must be a float32 tensor of shape {(nv,)}")
        if not result.is_contiguous():
            raise ValueError("result must be contiguous")

    ncp = _pow2(nc)
    nvp = _pow2(nv)
    # The 3-D broadcast tile is NCP*NCP*NVP elements; above 64 KiB it overflows
    # the UB, so fall back to the Cube path (which needs M, N, K >= 16).
    use_dot = (ncp * ncp * nvp > 16384) and ncp >= 16 and nvp >= 16

    ipm_kernel[(1,)](A, b, c, result, num_iters, nc, nv, ncp, nvp, use_dot)
    return result


# ---------------------------------------------------------------------------
# CANN 9.2: same algorithm, unrolled elimination sweeps and a cached launcher.
# ---------------------------------------------------------------------------


@triton.jit
def ipm_kernel_cann92(
    A_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    NUM_ITERS: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    NCP: tl.constexpr,
    NVP: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    oc = tl.arange(0, NCP).to(tl.int32)
    ov = tl.arange(0, NVP).to(tl.int32)

    m_a = (oc[:, None] < NC) & (ov[None, :] < NV)
    m_at = (ov[:, None] < NV) & (oc[None, :] < NC)

    # A (NC, NV) row-major plus its transposed view. ``at_t`` is materialised
    # once so that every A-side contraction can reduce along the contiguous
    # (axis=1) direction instead of a strided axis=0 sweep.
    a_t = tl.load(A_ptr + oc[:, None] * NV + ov[None, :], mask=m_a, other=0.0)
    at_t = tl.load(A_ptr + oc[None, :] * NV + ov[:, None], mask=m_at, other=0.0)
    b_t = tl.load(b_ptr + oc, mask=oc < NC, other=0.0)
    c_t = tl.load(c_ptr + ov, mask=ov < NV, other=0.0)

    eye = tl.where((oc[:, None] == oc[None, :]) & (oc[:, None] < NC), 1.0, 0.0)
    lowm = tl.where(oc[:, None] >= oc[None, :], 1.0, 0.0)

    # fp32 lane masks (0/1). Multiplying by these is vector-unit work on
    # Ascend; ``tl.where(<int compare>, <data tile>, 0.0)`` scalarises the whole
    # select, so it is avoided everywhere on the hot path.
    xmask = tl.where(ov < NV, 1.0, 0.0)
    negbig = tl.where(ov < NV, 0.0, -3.0e38)
    lmask = tl.where(ov == NV - 1, 1.0, 0.0)

    x = xmask
    d_max = 0.0

    for _ in range(NUM_ITERS):
        x2 = x * x
        ax2 = a_t * x2[None, :]
        if USE_DOT:
            ax2a = tl.dot(ax2, at_t) + 1e-12 * eye
        else:
            ax2a = tl.sum(ax2[:, None, :] * a_t[None, :, :], axis=2) + 1e-12 * eye
        ax2c = tl.sum(ax2 * c_t[None, :], axis=1)

        # ---- forward: in-place LDL^T elimination of the symmetric KKT matrix ----
        # Only the lower triangle is ever read back, so ``a`` starts as the
        # masked lower triangle and step k drops column k exactly, folding the
        # Schur complement and the L[:, k] write-back into one rank-1 update:
        #   * columns j != k get the usual  a[:, j] - L[:, k] * a[k, j],
        #   * column k is rewritten to L[:, k] via tvec[k] == 1.
        a = ax2a * lowm
        dinv = tl.zeros([NCP], dtype=tl.float32) + 1.0
        z = ax2c
        for k in tl.static_range(NC):
            mcol = tl.where(oc[None, :] == k, 1.0, 0.0)  # [1, NCP]
            ncol = tl.where(oc[None, :] == k, 0.0, 1.0)  # [1, NCP]
            mrow = tl.where(oc == k, 1.0, 0.0)  # [NCP]
            nrow = tl.where(oc == k, 0.0, 1.0)  # [NCP]

            cmat = a * mcol  # only column k survives
            col = tl.sum(cmat, axis=1)  # a[:, k]
            dk = tl.sum(col * mrow)  # a[k, k]
            inv = 1.0 / tl.maximum(dk, 1e-12)

            lk = (col * inv) * nrow  # L[:, k], unit diagonal implied
            tvec = mcol - col * ncol  # [1, NCP]
            a = (a - cmat) + lk[:, None] * tvec

            dinv = dinv * nrow + inv * mrow
            zi = tl.sum(z * mrow)  # z[k]
            z = z - lk * zi

        # ---- backward: L^T delta = D^-1 z  (row sweep) ----
        # The forward sweep no longer keeps the strictly upper triangle at zero
        # (it is never read there), so each extracted row is masked to j <= i.
        at = tl.trans(a)
        v = z * dinv
        for i in tl.static_range(NC - 1, -1, -1):
            mcol = tl.where(oc[None, :] == i, 1.0, 0.0)
            mrow = tl.where(oc == i, 1.0, 0.0)
            crow = tl.where(oc <= i, 1.0, 0.0)  # [NCP]
            ri = (tl.sum(at * mcol, axis=1)) * crow  # L[i, :]
            vi = tl.sum(v * mrow)
            v = v - ri * vi
        delta = v

        r = tl.sum(at_t * delta[None, :], axis=1)
        d = x * (c_t - r)
        d_max = tl.max(d * xmask + negbig)
        alpha = tl.where(d_max > 1e-9, 0.999 / d_max, 1.0)
        x = x * (1.0 - alpha * d)

    # ---- convergence test (mirrors the reference exactly) ----
    ax = tl.sum(a_t * x[None, :], axis=1)
    resid = tl.max(tl.abs(ax - b_t))
    x_last = tl.sum(x * lmask)
    ok = (d_max < 0.1) & (x_last >= 0.0) & (x_last < 1e-4) & (resid < 0.05)
    out_v = tl.where(ok, x, 0.5)
    tl.store(out_ptr + ov, out_v, mask=ov < NV)


def solve_ipm_cann92(A, b, c, num_iters=5, result=None):
    """CANN 9.2 kernel; identical contract to ``solve_ipm``."""
    if A.dim() != 2 or A.dtype != torch.float32:
        raise TypeError("A must be a 2-D float32 tensor")
    if b.dim() != 1 or b.dtype != torch.float32:
        raise TypeError("b must be a 1-D float32 tensor")
    if c.dim() != 1 or c.dtype != torch.float32:
        raise TypeError("c must be a 1-D float32 tensor")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous")
    if not b.is_contiguous():
        raise ValueError("b must be contiguous")
    if not c.is_contiguous():
        raise ValueError("c must be contiguous")
    if isinstance(num_iters, bool) or not isinstance(num_iters, int):
        raise TypeError("num_iters must be an integer")
    if num_iters < 0:
        raise ValueError("num_iters must be nonnegative")

    nc, nv = A.shape
    if nc < 1 or nv < 1:
        raise ValueError(f"A must be non-empty, got shape {(nc, nv)}")
    if b.shape != (nc,):
        raise ValueError(f"b must have shape {(nc,)}, got {tuple(b.shape)}")
    if c.shape != (nv,):
        raise ValueError(f"c must have shape {(nv,)}, got {tuple(c.shape)}")

    if result is None:
        result = torch.empty(nv, dtype=torch.float32, device=A.device)
    else:
        if result.shape != (nv,) or result.dtype != torch.float32:
            raise TypeError(f"result must be a float32 tensor of shape {(nv,)}")
        if not result.is_contiguous():
            raise ValueError("result must be contiguous")

    ncp = _pow2(nc)
    nvp = _pow2(nv)
    # The 3-D broadcast tile is NCP*NCP*NVP elements; above 64 KiB it overflows
    # the UB, so fall back to the Cube path (which needs M, N, K >= 16).
    use_dot = (ncp * ncp * nvp > 16384) and ncp >= 16 and nvp >= 16

    ipm_kernel_cann92[(1,)](A, b, c, result, num_iters, nc, nv, ncp, nvp, use_dot)
    return result
