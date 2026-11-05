"""Keep upstream precision tests limited to their supported hardware."""

import pytest


@pytest.fixture(autouse=True)
def require_ascend950(configure_npu):
    import torch

    pytest.importorskip("cannbotdsl")
    # configure_npu selects/restores the requested card; goldens stay on CPU.
    torch.set_default_device("cpu")
    name = torch.npu.get_device_name(torch.npu.current_device())
    if not name.startswith(("Ascend950PR", "Ascend950DT")):
        pytest.skip(f"Ascend 950 is required; found {name}")
