"""Regression checks for fused shared-memory tensor construction."""

import pytest
import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import fused_shm


def test_unsupported_dtype_fails_before_native_allocation(monkeypatch):
    calls = []

    def allocate(*args):
        calls.append(args)
        raise AssertionError("native allocator must not run for unsupported dtype")

    monkeypatch.setattr(
        torch.ops.npu, "fused_shm_allocator_create_and_register", allocate
    )

    with pytest.raises(TypeError, match="unsupported shm tensor dtype"):
        fused_shm.create_fused_shm_tensor((16,), torch.complex64)

    assert calls == []
