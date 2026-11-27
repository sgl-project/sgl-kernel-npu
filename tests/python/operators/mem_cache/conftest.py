"""Device checks for MemFabric integration tests."""

import pytest

ACL_DEV_ATTR_MAINBOARD_ID = 407
MF_URMA_UNSUPPORTED_BOARD_ID = 108  # MAIN_BOARD_ID_CARD_4PMESH


@pytest.fixture
def mf_local_device(configure_npu):
    import torch
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    device_id = torch.npu.current_device()
    if get_hardware().generation != Generation.ARCH35:
        return device_id

    try:
        import acl

        board_id, status = acl.rt.get_device_info(device_id, ACL_DEV_ATTR_MAINBOARD_ID)
    except (AttributeError, ImportError, RuntimeError):
        return device_id

    # PR #1082 could not initialize a MemFabric LOCAL pool on this board.
    if status == 0 and board_id == MF_URMA_UNSUPPORTED_BOARD_ID:
        pytest.skip("MemFabric LOCAL Host mapping is unsupported on Ascend 950 board 108")
    return device_id
