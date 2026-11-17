"""Distributed tests manage implementation pairs and performance in workers."""

import pytest


@pytest.fixture(autouse=True)
def configure_public_interface_test(request):
    # Override the single-worker executor context from tests/python/conftest.py.
    # In particular, do not advertise an unused auto-vs-torch_native impl pair.
    request.getfixturevalue("configure_npu")
    yield
