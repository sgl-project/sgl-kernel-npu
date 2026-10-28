"""Keep CPU reference tensors on CPU in the focused sampling tests."""

import pytest
import torch


@pytest.fixture
def impl_pair(request):
    # unittest cases do not participate in pytest_generate_tests parametrization.
    return request.config.getoption("impl"), request.config.getoption("impl_ref")


@pytest.fixture(autouse=True)
def cpu_reference_device():
    # The parent public-interface fixture defaults to NPU. These focused tests
    # explicitly move kernel inputs to NPU and construct CPU golden tensors.
    with torch.device("cpu"):
        yield
