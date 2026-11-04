"""A collected case makes hardware-only suite skips explicit to pytest."""


def test_ascend950_environment_is_available():
    # The autouse fixture in conftest.py skips before reaching this assertion
    # unless a supported Ascend 950 environment is active.
    assert True
