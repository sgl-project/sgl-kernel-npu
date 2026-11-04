"""Keep upstream precision tests limited to their supported hardware."""

import pytest


@pytest.fixture(autouse=True)
def require_ascend950():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("cannbotdsl")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    name = torch.npu.get_device_name(torch.npu.current_device())
    if not name.startswith(("Ascend950PR", "Ascend950DT")):
        pytest.skip(f"Ascend 950 is required; found {name}")
