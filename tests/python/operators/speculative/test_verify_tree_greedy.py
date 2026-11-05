"""Greedy chain verification through the public operator test framework.

Kernel and legacy fixture sources are recorded in the implementation modules and
in tests/python/sgl_kernel_npu/test_verify_tree.py. The old general-tree case tests
only the existing native implementation; it is not supported by this Triton kernel.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# SGLang snapshot: 77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc.
# eagle_utils.py:733-745 produces target IDs before NPU greedy verification;
# kernels/ops/speculative/spec_tree.py:523-536 imports the original NPU kernel.
# Model B/D/S coverage reuses the deployment choices confirmed for these NEXTN
# models. Both NPU CI recipes themselves use ordinary decoding, without NEXTN.
# V is only a token-ID generation range, not a dimension passed to this operator.
model_cases = {
    # B: test/registered/npu/accuracy/glm4_7_flash/test_npu_glm4_7_flash_1p_gsm8k.py
    # D=S=4: srt/arg_groups/speculative_hook.py default steps=3, topk=1.
    # V: https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json
    "glm-4-7-flash": [(batch, 4, 4, 154880) for batch in (1, 2, 4, 8, 16, 32)],
    # B: test/registered/npu/accuracy/qwen3_5_9b/test_npu_qwen3_5_9b_bf16_1p_gsm8k.py
    # Its B=16 is documented; B=1/2/4/8 are confirmed supplementary coverage.
    # D=S=4: test/registered/spec/eagle/test_eagle_reject_sampling.py supplies the
    # NEXTN dimensions; greedy verification uses target IDs instead of probabilities.
    # V: https://huggingface.co/Qwen/Qwen3.5-9B/blob/21eca8a083a2121a92fba681f4a7c72cf20ff1a7/config.json
    "qwen3-5-9b": [(batch, 4, 4, 248320) for batch in (1, 2, 4, 8, 16)],
}
perf_shape = [
    (1, 4, 4, 154880),
    (32, 4, 4, 154880),
    (1, 4, 4, 248320),
    (16, 4, 4, 248320),
]


def generate_test_data(shape, mode, dtype=torch.int64):
    """Synthetic token IDs with canonical slots and a known first mismatch."""
    batch, width, depth, vocab = shape
    assert depth == width, "the migrated Triton kernel requires S=D"
    candidates = torch.arange(batch * width, dtype=dtype).reshape(batch, width) % vocab
    slots = torch.arange(batch * width, dtype=dtype).reshape(batch, width)
    children = torch.arange(1, width + 1, dtype=dtype).repeat(batch, 1)
    children[:, -1] = -1
    siblings = torch.full_like(children, -1)
    target = torch.empty_like(candidates)
    target[:, :-1] = candidates[:, 1:]
    target[:, -1] = vocab - 1
    if mode != "all-accepted":
        row = {"reject-first": 0, "reject-partial": 1}[mode]
        target[:, row] = (candidates[:, row + 1] + 1) % vocab
    return dict(
        candidates=candidates,
        retrive_index=slots,
        retrive_next_token=children,
        retrive_next_sibling=siblings,
        target_predict=target,
    )


def _output_buffers(inputs, depth=None, predict_fill=-1):
    candidates = inputs["candidates"]
    if depth is None:
        depth = candidates.shape[1]
    return dict(
        predicts=torch.full(
            (candidates.numel(),),
            predict_fill,
            dtype=torch.int32,
            device=candidates.device,
        ),
        accept_index=torch.full(
            (candidates.shape[0], depth),
            -1,
            dtype=torch.int32,
            device=candidates.device,
        ),
        accept_token_num=torch.zeros(
            candidates.shape[0], dtype=torch.int32, device=candidates.device
        ),
    )


def _check_case(inputs, *, predict_fill=-1, expected_accept=None, enable_perf=False):
    from sgl_kernel_npu.operators.speculative import verify_tree_greedy

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(verify_tree_greedy, impl)
    else:
        require_distinct_impls(verify_tree_greedy, impl, impl_ref)
    outputs = _output_buffers(inputs, predict_fill=predict_fill)
    if not skip_accuracy:
        snapshots = {name: value.clone() for name, value in inputs.items()}
        reference = _output_buffers(inputs, predict_fill=predict_fill)
        reference_result = verify_tree_greedy(**reference, **inputs, impl=impl_ref)
        assert reference_result is None
        for name, value in inputs.items():
            assert torch.equal(value, snapshots[name]), f"reference modified {name}"
        if expected_accept is not None:
            assert_accuracy(
                reference["accept_token_num"],
                torch.full_like(reference["accept_token_num"], expected_accept),
            )
        result = verify_tree_greedy(**outputs, **inputs, impl=impl)
        assert result is None
        assert_accuracy(outputs, reference)
        for name, value in inputs.items():
            assert torch.equal(
                value, snapshots[name]
            ), f"implementation modified {name}"
        del snapshots, reference
    executor = make_executor(
        verify_tree_greedy,
        kwargs={**outputs, **inputs},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=enable_perf)


def generate_legacy_simple_data():
    """Original int32 B=2/D=2 fixture, using the configured default NPU."""
    candidates = torch.tensor([[10375, 28], [223, 15098]], dtype=torch.int32)
    target_predict = torch.tensor([[28, 334], [15098, 18]], dtype=torch.int32)
    retrive_next_sibling = torch.tensor([[-1, -1], [-1, -1]], dtype=torch.int32)
    retrive_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    retrive_next_token = torch.tensor([[1, -1], [1, -1]], dtype=torch.int32)
    return dict(
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


def generate_legacy_general_tree_data():
    """Original B=2/D=6/S=4 fixture for native-only regression."""
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
    )
    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i][j]) < 10:
                target_logits[i][j][18] = 10
    target_predict = torch.argmax(target_logits, dim=-1)
    return dict(
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


def test_verify_tree_greedy_legacy_simple():
    _check_case(generate_legacy_simple_data(), predict_fill=0, expected_accept=1)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(shape, id=model)
        for model, shapes in model_cases.items()
        for shape in shapes
    ],
)
@pytest.mark.parametrize("mode", ["all-accepted", "reject-first", "reject-partial"])
def test_verify_tree_greedy_model_shapes(shape, mode):
    expected = {"all-accepted": shape[1] - 1, "reject-first": 0, "reject-partial": 1}[
        mode
    ]
    _check_case(
        generate_test_data(shape, mode),
        expected_accept=expected,
        enable_perf=shape in perf_shape,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "shape,mode,accepted",
    [
        pytest.param((1, 1, 1, 20), "all-accepted", 0, id="root-only"),
        pytest.param((2, 3, 3, 20), "reject-partial", 1, id="padded-width-three"),
        pytest.param((2, 5, 5, 20), "reject-first", 0, id="padded-width-five"),
    ],
)
def test_verify_tree_greedy_boundaries(shape, mode, accepted, dtype):
    _check_case(generate_test_data(shape, mode, dtype), expected_accept=accepted)


@pytest.mark.extra_checks
def test_existing_native_general_tree_case():
    # The original test calls only verify_tree_greedy_native for this S<D tree.
    # Preserve its values and expected results without launching the linear-only
    # Triton kernel, whose accept_index row stride is D rather than S.
    if bool(get_skip_acc_check()):
        return
    from sgl_kernel_npu.operators.speculative import verify_tree_greedy

    require_test_impl(verify_tree_greedy, "torch_native")
    inputs = generate_legacy_general_tree_data()
    snapshots = {name: value.clone() for name, value in inputs.items()}
    outputs = _output_buffers(inputs, depth=4)
    result = verify_tree_greedy(**outputs, **inputs, impl="torch_native")
    assert result is None
    expected = dict(
        predicts=torch.tensor(
            [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18], dtype=torch.int32
        ),
        accept_index=torch.tensor([[0, 3, 4, 5], [6, 10, 11, -1]], dtype=torch.int32),
        accept_token_num=torch.tensor([3, 2], dtype=torch.int32),
    )
    assert_accuracy(outputs, expected)
    for name, value in inputs.items():
        assert torch.equal(value, snapshots[name]), f"native modified {name}"
