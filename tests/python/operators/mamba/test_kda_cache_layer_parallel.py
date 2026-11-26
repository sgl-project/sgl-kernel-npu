"""Check layer-parallel cache updates, including in-place and stride semantics."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.common import get_test_impls, require_distinct_impls


def generate_data(op, shape, batch, dtype, transposed, device):
    torch.manual_seed(42)
    layers, *dims = shape
    pool, drafts = 2 * batch + 3, 4
    storage_shape = (layers, pool, *dims)
    if transposed:
        storage_shape = (*storage_shape[:-2], storage_shape[-1], storage_shape[-2])
    state = torch.randn(storage_shape, dtype=dtype, device=device)
    if transposed:
        state = state.transpose(-1, -2)
    slots = list(reversed(range(1, batch + 1)))
    steps = [(0, -1, 1, drafts - 1)[i % 4] for i in range(batch)]
    indices = torch.tensor(slots, dtype=torch.int32, device=device)
    step_indices = torch.tensor(steps, dtype=torch.int32, device=device)
    expected = state.clone()
    if op == "move":
        source = torch.randn(
            layers, max(batch, 1), drafts, *dims, dtype=dtype, device=device
        )
        src_indices = torch.arange(batch, dtype=torch.int32, device=device)
        args = (source, indices, src_indices, step_indices)
    else:
        args = (indices, step_indices, drafts)

    # Independent reference: skipped requests and unselected pool slots stay intact.
    for src, (dst, step) in enumerate(zip(slots, steps)):
        if step < 0:
            continue
        if op == "move":
            expected[:, dst] = source[:, src, step]
        else:
            shift = drafts - 1 - step
            if shift > 0:
                expected[:, dst, shift:] = expected[:, dst, :-shift].clone()
    return state, args, expected


def check_case(op, shape, batch, dtype, transposed):
    from sgl_kernel_npu.operators.mamba import (
        conv_state_rollback,
        move_intermediate_cache_kda,
    )

    operator = move_intermediate_cache_kda if op == "move" else conv_state_rollback
    impl, impl_ref = get_test_impls()
    require_distinct_impls(operator, impl, impl_ref)
    state, args, expected = generate_data(
        op, shape, batch, dtype, transposed, f"npu:{torch.npu.current_device()}"
    )
    for implementation in (impl_ref, impl):
        # Each implementation receives the same initial values and layout.
        actual = state.clone()
        strides = actual.stride()
        result = operator(actual, *args, impl=implementation)
        if op == "conv" and batch == 0:
            assert result is None
        else:
            assert result.data_ptr() == actual.data_ptr()
            assert result.stride() == strides
        assert actual.stride() == state.stride()
        # Bound assertion temporaries for model shapes; inspect the whole pool.
        for actual_layer, expected_layer in zip(actual, expected):
            for start in range(0, state.shape[1], 8):
                torch.testing.assert_close(
                    actual_layer[start : start + 8],
                    expected_layer[start : start + 8],
                    atol=0,
                    rtol=0,
                )


@pytest.mark.parametrize("op", ["move", "conv"])
@pytest.mark.parametrize(
    "layers,batch", [(1, 1), (4, 0), (4, 1), (32, 1), (4, 8), (4, 128)]
)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kda_cache_layer_parallel(op, layers, batch, transposed, dtype):
    # Unaligned V/K dimensions exercise partial tiles; batch 8 mixes all steps.
    shape = (layers, 2, 65, 19) if op == "move" else (layers, 6, 32)
    check_case(op, shape, batch, dtype, transposed)


@pytest.mark.parametrize(
    "op,shape,dtype,transposed",
    [
        ("move", (69, 6, 128, 128), torch.float32, True),
        ("conv", (69, 6, 2304), torch.bfloat16, False),
    ],
)
def test_kimi31_cache_shape(op, shape, dtype, transposed):
    # Kimi K3.1: 69 KDA layers, 96 heads, head dim 128, conv kernel 4.
    # Test choices: one TP=16 shard (6 heads), four drafts, batch 8.
    # Conv window=(4-1)+(4-1)=6; channels=3*6*128=2304.
    check_case(op, shape, 8, dtype, transposed)


def test_legacy_entry_points_share_optimized_implementation():
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
