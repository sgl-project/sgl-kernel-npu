# Verify that Dynamo can capture the real NPU core-count query.
def test_get_device_properties_fullgraph(npu_runtime, package_factory):
    import torch_npu

    from tests.python.utils.accuracy import assert_accuracy

    torch = npu_runtime.torch
    triton_utils = package_factory().import_module(
        "implementation.triton.utils.triton_utils"
    )
    properties = torch_npu.npu.get_device_properties(torch_npu.npu.current_device())

    def add_core_counts(x):
        cube_cores, vector_cores = triton_utils.get_device_properties()
        return x + cube_cores, x + vector_cores

    x = torch.zeros(1, dtype=torch.int32, device="npu")
    compiled = torch.compile(add_core_counts, backend="eager", fullgraph=True)
    assert_accuracy(
        compiled(x), (x + properties.cube_core_num, x + properties.vector_core_num)
    )
