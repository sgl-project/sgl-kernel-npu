"""Unit tests for the fused sigmoid-gate-multiply Triton kernels.

Covers both variants in ``sgl_kernel_npu/activation/fused_sigmoid_mul.py``:

- ``fused_sigmoid_mul``: element-wise ``x * sigmoid(gate)`` for same-shape
  inputs of any rank.
- ``fused_sigmoid_mul_broadcast``: ``x * sigmoid(gate)`` for 2-D ``x`` with a
  per-row scalar gate of shape ``(N,)`` or ``(N, 1)``.

In-repo Python sources are preferred over the installed package so the test
exercises the current tree; compiled ops still come from the installed
``sgl_kernel_npu`` package.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# Installed package: loads the compiled ops and provides shared utils.
import sgl_kernel_npu  # noqa: E402

_repo_pkg = os.path.join(REPO_ROOT, "python", "sgl_kernel_npu", "sgl_kernel_npu")
if os.path.isdir(_repo_pkg) and _repo_pkg not in sgl_kernel_npu.__path__:
    sgl_kernel_npu.__path__.insert(0, _repo_pkg)

from sgl_kernel_npu.activation.fused_sigmoid_mul import (  # noqa: E402
    fused_sigmoid_mul,
    fused_sigmoid_mul_broadcast,
)

# Fail loudly if the submodule did not come from the in-repo sources.
if os.path.isdir(_repo_pkg):
    _mod = sys.modules["sgl_kernel_npu.activation.fused_sigmoid_mul"]
    assert _mod.__file__.startswith(_repo_pkg), (
        f"expected in-repo module, got {_mod.__file__}"
    )

DEVICE = "npu:0"

_TOLS = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=1.6e-2, atol=1.6e-2),
}

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)


def _check(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> None:
    torch.testing.assert_close(out.float(), ref.float(), **_TOLS[dtype])


def test_fused_sigmoid_mul_elementwise():
    for dtype in _DTYPES:
        for shape in [(1, 4096), (16, 4096), (32, 3584), (128, 5120), (4, 16, 256)]:
            x = torch.randn(shape, dtype=dtype, device=DEVICE)
            gate = torch.randn(shape, dtype=dtype, device=DEVICE)
            out = fused_sigmoid_mul(x, gate)
            assert out.shape == x.shape
            assert out.dtype == x.dtype
            _check(out, _reference(x, gate), dtype)


def test_fused_sigmoid_mul_non_contiguous():
    # Strided views must produce the same result as contiguous inputs.
    x = torch.randn(32, 8192, dtype=torch.bfloat16, device=DEVICE)[:, ::2]
    gate = torch.randn(32, 8192, dtype=torch.bfloat16, device=DEVICE)[:, ::2]
    out = fused_sigmoid_mul(x, gate)
    _check(out, _reference(x, gate), torch.bfloat16)


def test_fused_sigmoid_mul_empty():
    x = torch.empty(0, 4096, dtype=torch.bfloat16, device=DEVICE)
    out = fused_sigmoid_mul(x, x)
    assert out.shape == x.shape


def test_fused_sigmoid_mul_shape_mismatch():
    x = torch.randn(32, 4096, device=DEVICE)
    gate = torch.randn(32, 2048, device=DEVICE)
    with pytest.raises(ValueError):
        fused_sigmoid_mul(x, gate)


def test_fused_sigmoid_mul_broadcast():
    for dtype in _DTYPES:
        for n, d in [(1, 4096), (32, 4096), (128, 5120)]:
            x = torch.randn(n, d, dtype=dtype, device=DEVICE)
            for gate in (
                torch.randn(n, dtype=dtype, device=DEVICE),
                torch.randn(n, 1, dtype=dtype, device=DEVICE),
            ):
                out = fused_sigmoid_mul_broadcast(x, gate)
                assert out.shape == x.shape
                assert out.dtype == x.dtype
                # The gate is per-row; the reference broadcasts it explicitly.
                _check(out, _reference(x, gate.reshape(n, 1)), dtype)


def test_fused_sigmoid_mul_broadcast_bad_gate():
    x = torch.randn(32, 4096, device=DEVICE)
    # 2-D gate must have exactly one column.
    with pytest.raises(ValueError):
        fused_sigmoid_mul_broadcast(x, torch.randn(32, 2, device=DEVICE))
    # 1-D gate must match the row count.
    with pytest.raises(ValueError):
        fused_sigmoid_mul_broadcast(x, torch.randn(64, device=DEVICE))
    # x must be 2-D.
    with pytest.raises(ValueError):
        fused_sigmoid_mul_broadcast(
            torch.randn(4, 16, 256, device=DEVICE), torch.randn(4, device=DEVICE)
        )


if __name__ == "__main__":
    test_fused_sigmoid_mul_elementwise()
    test_fused_sigmoid_mul_non_contiguous()
    test_fused_sigmoid_mul_empty()
    test_fused_sigmoid_mul_shape_mismatch()
    test_fused_sigmoid_mul_broadcast()
    test_fused_sigmoid_mul_broadcast_bad_gate()
    print("All fused_sigmoid_mul tests passed.")
