"""Opt-in CANN version fixtures for operator tests."""

import pytest


@pytest.fixture(scope="module")
def require_cann_9_2_0_or_newer():
    from packaging.version import Version
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version

    version = detect_cann_version()
    if Version(version) < Version("9.2.0"):
        pytest.skip(
            f"The released CANN backend requires CANN >= 9.2.0 (found {version})"
        )


@pytest.fixture(scope="module", autouse=True)
def require_selected_cann_backend(request):
    """Gate an explicitly selected 9.2 backend; auto retains legacy coverage."""
    implementations = [request.config.getoption("impl")]
    if request.config.getoption("skip_acc_check") is not True:
        implementations.append(request.config.getoption("impl_ref"))
    if "cann_ops_transformers_cann920" in implementations:
        request.getfixturevalue("require_cann_9_2_0_or_newer")
