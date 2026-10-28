"""Local opt-in for supplementary RoPE checks."""

import pytest


def pytest_addoption(parser):
    parser.getgroup("RoPE checks").addoption(
        "--run-rope-extra-checks",
        action="store_true",
        default=False,
        help="Run supplementary RoPE output-contract and invalid-input checks.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "rope_extra_checks: supplementary RoPE checks, skipped by default"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("run_rope_extra_checks"):
        return
    skip = pytest.mark.skip(reason="Enable with --run-rope-extra-checks")
    for item in items:
        if item.get_closest_marker("rope_extra_checks") is not None:
            item.add_marker(skip)
