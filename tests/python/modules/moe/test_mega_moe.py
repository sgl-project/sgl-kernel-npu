"""Contract tests for the MegaMoE public registration.

These tests exercise registry bookkeeping and lazy loading.  Numerical CANN
coverage belongs to the provider's NPU integration tests because this checkout
does not provide an NPU device or the compiled MC2 extension.
"""

from types import ModuleType
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_mega_moe_registration_is_complete():
    from sgl_kernel_npu.operators import list_impls
    from sgl_kernel_npu.modules.moe import mega_moe

    assert list_impls(mega_moe) == ["auto", "cann"]


def test_mega_moe_is_registered_for_both_generations():
    from sgl_kernel_npu.modules.moe import mega_moe
    from sgl_kernel_npu.operators.utils import generation, registry

    assert set(generation.Generation) == {
        generation.Generation.ARCH22,
        generation.Generation.ARCH35,
    }
    assert registry._UNSUPPORTED_IMPLS[
        ("moe.mega_moe", "auto")
    ] == frozenset()
    assert registry._UNSUPPORTED_IMPLS[
        ("moe.mega_moe", "cann")
    ] == frozenset()
    assert registry.get_op_name(mega_moe) == "moe.mega_moe"


def test_mega_moe_backend_is_loaded_only_on_call():
    from sgl_kernel_npu.modules.moe import mega_moe

    fake_result = object()
    fake_backend = ModuleType("cann_ops_transformer.ops")
    fake_backend.mega_moe = lambda *args, **kwargs: fake_result
    fake_backend.get_symm_buffer_for_mega_moe = lambda *args, **kwargs: "buffer"
    fake_parent = ModuleType("cann_ops_transformer")
    fake_parent.ops = fake_backend
    with patch.dict(
        "sys.modules",
        {"cann_ops_transformer": fake_parent, "cann_ops_transformer.ops": fake_backend},
    ):
        assert mega_moe("x", impl="cann") is fake_result


def test_mega_moe_unknown_impl_fails_before_backend_load():
    from sgl_kernel_npu.modules.moe import mega_moe
    from sgl_kernel_npu.operators.utils.registry import ImplementationNotFoundError

    with pytest.raises(ImplementationNotFoundError):
        mega_moe("x", impl="missing")
