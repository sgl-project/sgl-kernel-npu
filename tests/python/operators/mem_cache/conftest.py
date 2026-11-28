"""Host architecture checks for MemFabric integration tests."""

import platform

import pytest


@pytest.fixture
def mf_local_device(configure_npu):
    if platform.machine().lower() in ("x86_64", "amd64"):
        pytest.skip("MemFabric LOCAL KV transfer is unsupported on x86 hosts")

    import torch

    return torch.npu.current_device()
