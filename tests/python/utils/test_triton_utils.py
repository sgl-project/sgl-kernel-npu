import pytest


@pytest.mark.extra_checks
@pytest.mark.usefixtures("configure_npu")
def test_get_device_properties_fullgraph():
    import torch
    import torch_npu
    from sgl_kernel_npu.implementation.triton.utils.triton_utils import (
        get_device_properties,
    )

    from tests.python.utils.accuracy import assert_accuracy

    properties = torch_npu.npu.get_device_properties(torch_npu.npu.current_device())
    get_device_properties.cache_clear()

    def add_core_counts(x):
        cube_cores, vector_cores = get_device_properties()
        return x + cube_cores, x + vector_cores

    x = torch.zeros(1, dtype=torch.int32, device="npu")
    compiled = torch.compile(add_core_counts, backend="eager", fullgraph=True)
    assert_accuracy(
        compiled(x), (x + properties.cube_core_num, x + properties.vector_core_num)
    )
