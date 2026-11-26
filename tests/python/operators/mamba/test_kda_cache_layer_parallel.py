"""Exact, in-place KDA cache checks through the mainline public API.

Performance uses the standalone A/B runner, which resets each destination outside
the event interval. A repeatedly invoked shared executor would otherwise time an
already-mutated rollback buffer. See benchmark/kda_cache_layer_parallel/README.md.
"""

import importlib
import importlib.util
import inspect
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.common import get_test_impls, require_distinct_impls

REPO = Path(__file__).resolve().parents[4]
PACKAGE = REPO / "python/sgl_kernel_npu/sgl_kernel_npu"
_spec = importlib.util.spec_from_file_location(
    "kda_cache_layer_benchmark", REPO / "benchmark/kda_cache_layer_parallel/bench.py"
)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


@pytest.fixture
def operators():
    from sgl_kernel_npu.operators.mamba import (
        conv_state_rollback,
        move_intermediate_cache_kda,
    )

    backend = importlib.import_module(
        "sgl_kernel_npu.implementation.triton.mamba.mamba_state_update"
    )
    assert (
        Path(backend.__file__).resolve()
        == (PACKAGE / "implementation/triton/mamba/mamba_state_update.py").resolve()
    ), "Install this checkout before testing; the loaded backend is stale"
    public = dict(
        move_intermediate_cache_kda=move_intermediate_cache_kda,
        conv_state_rollback=conv_state_rollback,
    )
    impl, impl_ref = get_test_impls()
    for name, operator in public.items():
        assert (
            Path(inspect.getfile(operator)).resolve()
            == (PACKAGE / f"operators/mamba/{name}.py").resolve()
        ), "The public API must also come from this checkout"
        require_distinct_impls(operator, impl, impl_ref)

    def bind(implementation):
        return SimpleNamespace(
            **{name: partial(fn, impl=implementation) for name, fn in public.items()}
        )

    return f"npu:{torch.npu.current_device()}", bind(impl_ref), bind(impl)


@pytest.mark.parametrize(
    "op,layer_batch,transposed,dtype", list(bench.correctness_cases())
)
def test_kda_cache_layer_parallel(op, layer_batch, transposed, dtype, operators):
    device, reference, optimized = operators
    layers, batch = layer_batch
    shape = dict(
        layers=layers,
        drafts=4,
        heads=2,
        value_dim=65,
        key_dim=19,
        window=6,
        channels=32,
    )
    data = bench.generate_data(op, batch, shape, device, dtype, transposed, mixed=True)
    # Each public implementation gets its own reset buffer. Both must agree with
    # the independently constructed expected state, including untouched slots.
    bench.check_accuracy(data, reference, optimized)


@pytest.mark.parametrize("op,shape,dtype,transposed", bench.MODEL_CASES)
def test_kimi31_cache_shape(op, shape, dtype, transposed, operators):
    device, reference, optimized = operators
    data = bench.generate_data(op, 8, shape, device, dtype, transposed, mixed=True)
    bench.check_accuracy(data, reference, optimized)


def test_legacy_entry_points_share_optimized_implementation(operators):
    from sgl_kernel_npu.implementation.triton.mamba import mamba_state_update
    from sgl_kernel_npu.mamba import mamba_state_update_triton

    for name in (
        "move_cache_dynamic_last_kernel_h_block_kda",
        "move_intermediate_cache_kda",
        "_conv_state_rollback_kernel",
        "conv_state_rollback",
    ):
        assert getattr(mamba_state_update_triton, name) is getattr(
            mamba_state_update, name
        )
