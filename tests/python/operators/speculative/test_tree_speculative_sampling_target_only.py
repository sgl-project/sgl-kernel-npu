"""Original and model-shaped tree sampling through the public operator framework.

Input construction is copied from tests/python/sgl_kernel_npu/
test_tree_speculative_sampling_target_only.py at sgl-project/sgl-kernel-npu
commit 15ea25766eb0f9a30951d7d127af8600998272ea (#623).
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
from tests.python.utils.standards import AccuracyStandard

# The original general-tree test compares scratch with zero tolerance. Scratch
# is cleared and receives exact copies of FP32 target probabilities.
SCRATCH_STANDARD = {"torch.float32": AccuracyStandard(atol=0, rtol=0)}
THRESHOLDS = [
    pytest.param(1.0, 1.0, id="default-thresholds"),
    pytest.param(0.0, 0.0, id="zero-thresholds"),
    pytest.param(0.5, 0.8, id="relaxed-thresholds"),
]

# Source revision for the SGLang deployment and speculative configurations:
# https://github.com/sgl-project/sglang/tree/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc
# These are deployment-derived operator cases, rather than recorded kernel traces.
# The GLM/Qwen NPU CI configurations provide ordinary-decoding batch sizes.
# Combine them with NEXTN topk=1 chains and disable rejection sampling to exercise
# target-only sampling. Shapes are (batch, candidate slots, max tree depth, vocab).
model_cases = {
    # GLM NPU CI uses TP2, max-running-requests=32 and graph sizes 1/2/4/8/16/32.
    # test/registered/npu/accuracy/glm4_7_flash/test_npu_glm4_7_flash_1p_gsm8k.py
    # D=S=4 follows Glm4MoeLiteForCausalLM's default three-step NEXTN chain in
    # python/sglang/srt/arg_groups/speculative_hook.py
    # (_auto_choose_speculative_params).
    # V: https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json
    "glm-4-7-flash": [(batch, 4, 4, 154880) for batch in (1, 2, 4, 8, 16, 32)],
    # Qwen NPU CI uses TP2 and captures B=16. B=1/2/4/8 cover smaller workloads.
    # test/registered/npu/accuracy/qwen3_5_9b/test_npu_qwen3_5_9b_bf16_1p_gsm8k.py
    # D=S=4 comes from the CUDA NEXTN configuration; target-only uses its chain
    # dimensions with speculative-use-rejection-sampling disabled.
    # test/registered/spec/eagle/test_eagle_reject_sampling.py
    # V: https://huggingface.co/Qwen/Qwen3.5-9B/blob/21eca8a083a2121a92fba681f4a7c72cf20ff1a7/config.json
    "qwen3-5-9b": [(batch, 4, 4, 248320) for batch in (1, 2, 4, 8, 16)],
}

# Default profiling covers the smallest and largest batch of each model.
perf_shape = [
    (1, 4, 4, 154880),
    (32, 4, 4, 154880),
    (1, 4, 4, 248320),
    (16, 4, 4, 248320),
]


def _output_buffers(inputs, max_tree_depth):
    candidates = inputs["candidates"]
    return {
        "predicts": torch.full(
            (candidates.numel(),), -1, dtype=torch.int32, device=candidates.device
        ),
        "accept_index": torch.full(
            (candidates.shape[0], max_tree_depth),
            -1,
            dtype=torch.int32,
            device=candidates.device,
        ),
        "accept_token_num": torch.zeros(
            candidates.shape[0], dtype=torch.int32, device=candidates.device
        ),
        "draft_probs": torch.empty_like(inputs["target_probs"]),
    }


def _check_case(
    inputs,
    max_tree_depth,
    threshold_single=1.0,
    threshold_acc=1.0,
    *,
    enable_perf=False,
):
    from sgl_kernel_npu.operators.speculative import (
        tree_speculative_sampling_target_only,
    )

    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(tree_speculative_sampling_target_only, impl)
    else:
        require_distinct_impls(tree_speculative_sampling_target_only, impl, impl_ref)
    options = {
        "threshold_single": threshold_single,
        "threshold_acc": threshold_acc,
        "deterministic": True,
    }
    outputs = _output_buffers(inputs, max_tree_depth)
    if not skip_accuracy:
        snapshots = {name: value.clone() for name, value in inputs.items()}
        reference_outputs = _output_buffers(inputs, max_tree_depth)
        reference_result = tree_speculative_sampling_target_only(
            **reference_outputs, **inputs, **options, impl=impl_ref
        )
        assert reference_result is None
        for name, value in inputs.items():
            assert torch.equal(value, snapshots[name]), f"reference modified {name}"

        result = tree_speculative_sampling_target_only(
            **outputs, **inputs, **options, impl=impl
        )
        assert result is None
        assert_accuracy(outputs, reference_outputs, standard=SCRATCH_STANDARD)
        for name, value in inputs.items():
            assert torch.equal(
                value, snapshots[name]
            ), f"implementation modified {name}"

        del snapshots, reference_outputs

    executor = make_executor(
        tree_speculative_sampling_target_only,
        kwargs={**outputs, **inputs, **options},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    # Legacy accuracy cases remain accuracy-only unless the CLI requests profiling.
    # The wrapper's own scratch clearing remains within the measured operation.
    check_performance(executor, enable_perf=enable_perf)


def make_chain_indices(batch_size, num_draft_tokens, device):
    retrive_index = torch.arange(
        batch_size * num_draft_tokens, dtype=torch.int64, device=device
    ).view(batch_size, num_draft_tokens)
    retrive_next_token = torch.arange(
        1, num_draft_tokens + 1, dtype=torch.int64, device=device
    ).repeat(batch_size, 1)
    retrive_next_token[:, -1] = -1
    retrive_next_sibling = torch.full_like(retrive_next_token, -1)
    return retrive_index, retrive_next_token, retrive_next_sibling


def make_stable_chain_final_uniforms(
    candidates,
    uniform_samples,
    target_probs,
):
    """Choose final-sampling coins away from inverse-CDF boundaries."""
    batch_size, num_draft_tokens = candidates.shape
    final_uniforms = torch.empty(batch_size, dtype=torch.float32)

    for req_idx in range(batch_size):
        num_accepted = 0
        rejected_token = -1
        for step in range(1, num_draft_tokens):
            draft_token = int(candidates[req_idx, step])
            target_prob = float(target_probs[req_idx, step - 1, draft_token])
            if float(uniform_samples[req_idx, step - 1]) <= target_prob:
                num_accepted += 1
            else:
                rejected_token = draft_token
                break

        final_probs = target_probs[req_idx, num_accepted].double().clone()
        if rejected_token >= 0:
            final_probs[rejected_token] = 0.0

        sampled_token = int(final_probs.argmax())
        probability_sum = final_probs.sum()
        cdf_before = final_probs[:sampled_token].sum()
        cdf_midpoint = cdf_before + final_probs[sampled_token] * 0.5
        final_uniforms[req_idx] = (cdf_midpoint / probability_sum).float()

    return final_uniforms


def generate_general_tree_test_data():
    with torch.device("cpu"):
        candidates = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]], dtype=torch.int64
        )
        retrive_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int64
        )
        retrive_next_token = torch.tensor(
            [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
            dtype=torch.int64,
        )
        retrive_next_sibling = torch.tensor(
            [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
            dtype=torch.int64,
        )
        batch_size, num_draft_tokens = candidates.shape
        vocab_size = 20
        target_probs = torch.full(
            (batch_size, num_draft_tokens, vocab_size), 0.01, dtype=torch.float32
        )
        target_probs[0, 0, 1] = 0.12
        target_probs[0, 0, 3] = 0.72
        target_probs[0, 3, 4] = 0.82
        target_probs[0, 4, 5] = 0.75
        target_probs[1, 0, 11] = 0.68
        target_probs[1, 0, 8] = 0.14
        target_probs[1, 4, 12] = 0.77
        target_probs /= target_probs.sum(dim=-1, keepdim=True)
        uniforms = torch.tensor(
            [[0.55, 0.2, 0.8, 0.4, 0.3, 0.9], [0.6, 0.2, 0.8, 0.7, 0.3, 0.4]],
            dtype=torch.float32,
        )
        final_uniforms = torch.tensor([0.25, 0.75], dtype=torch.float32)
        inputs = {
            "candidates": candidates,
            "retrive_index": retrive_index,
            "retrive_next_token": retrive_next_token,
            "retrive_next_sibling": retrive_next_sibling,
            "uniform_samples": uniforms,
            "uniform_samples_for_final_sampling": final_uniforms,
            "target_probs": target_probs,
        }
    return {name: value.npu() for name, value in inputs.items()}


def generate_chain_test_data(batch_size, num_draft_tokens, vocab_size):
    with torch.device("cpu"):
        torch.manual_seed(20260717 + batch_size + num_draft_tokens + vocab_size)
        candidates = torch.randint(
            0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int64
        )
        logits = torch.randn(batch_size, num_draft_tokens, vocab_size)
        target_probs = torch.softmax(logits, dim=-1).float()

        # Give some draft tokens meaningful acceptance probability.
        for req_idx in range(batch_size):
            for step in range(1, num_draft_tokens):
                token = int(candidates[req_idx, step])
                target_probs[req_idx, step - 1] *= 0.35
                target_probs[req_idx, step - 1, token] += 0.65
                target_probs[req_idx, step - 1] /= target_probs[req_idx, step - 1].sum()

        uniform_samples = torch.rand(batch_size, num_draft_tokens)
        # A random coin can land within FP32 reduction error of a CDF boundary for
        # large vocabularies. Use the midpoint of a high-mass token's interval so
        # exact token equality tests the algorithm instead of reduction order.
        final_uniform_samples = make_stable_chain_final_uniforms(
            candidates,
            uniform_samples,
            target_probs,
        )
        retrive_index, retrive_next_token, retrive_next_sibling = make_chain_indices(
            batch_size, num_draft_tokens, "cpu"
        )
        inputs = {
            "candidates": candidates,
            "retrive_index": retrive_index,
            "retrive_next_token": retrive_next_token,
            "retrive_next_sibling": retrive_next_sibling,
            "uniform_samples": uniform_samples,
            "uniform_samples_for_final_sampling": final_uniform_samples,
            "target_probs": target_probs,
        }
    return {name: value.npu() for name, value in inputs.items()}


def generate_threshold_test_data():
    with torch.device("cpu"):
        batch_size, num_draft_tokens, vocab_size = 2, 4, 32
        candidates = torch.tensor([[0, 3, 4, 5], [0, 7, 8, 9]], dtype=torch.int64)
        target_probs = torch.full(
            (batch_size, num_draft_tokens, vocab_size), 1.0 / vocab_size
        )
        for req_idx in range(batch_size):
            for step in range(1, num_draft_tokens):
                token = int(candidates[req_idx, step])
                target_probs[req_idx, step - 1] *= 0.2
                target_probs[req_idx, step - 1, token] += 0.8
                target_probs[req_idx, step - 1] /= target_probs[req_idx, step - 1].sum()

        uniforms = torch.tensor([[0.1, 0.9, 0.2, 0.0], [0.7, 0.2, 0.95, 0.0]])
        final_uniforms = torch.tensor([0.25, 0.75])
        retrive_index, next_token, next_sibling = make_chain_indices(
            batch_size, num_draft_tokens, "cpu"
        )
        inputs = {
            "candidates": candidates,
            "retrive_index": retrive_index,
            "retrive_next_token": next_token,
            "retrive_next_sibling": next_sibling,
            "uniform_samples": uniforms,
            "uniform_samples_for_final_sampling": final_uniforms,
            "target_probs": target_probs,
        }
    return {name: value.npu() for name, value in inputs.items()}


@pytest.mark.parametrize("threshold_single,threshold_acc", THRESHOLDS)
def test_general_tree_matches_reference(threshold_single, threshold_acc):
    _check_case(
        generate_general_tree_test_data(),
        max_tree_depth=4,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            (batch, drafts, vocab), id=f"legacy-chain-b{batch}-d{drafts}-v{vocab}"
        )
        for batch in (1, 4, 17)
        for drafts in (2, 5)
        for vocab in (20, 32000, 151552)
    ],
)
def test_target_only_chain_matches_reference(shape):
    _check_case(generate_chain_test_data(*shape), max_tree_depth=shape[1])


@pytest.mark.parametrize("threshold_single,threshold_acc", THRESHOLDS)
def test_target_only_thresholds(threshold_single, threshold_acc):
    _check_case(
        generate_threshold_test_data(),
        max_tree_depth=4,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(shape, id=model)
        for model, shapes in model_cases.items()
        for shape in shapes
    ],
)
def test_tree_speculative_sampling_model_shapes(shape):
    batch, candidates, max_tree_depth, vocab = shape
    inputs = generate_chain_test_data(batch, candidates, vocab)
    _check_case(inputs, max_tree_depth, enable_perf=shape in perf_shape)
