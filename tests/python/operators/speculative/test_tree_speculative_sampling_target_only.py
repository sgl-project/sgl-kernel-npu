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

# Core model coverage pairs single-request GLM with batched Qwen. The default
# general-tree case below also preserves branching and S<D coverage.
default_shapes = [(1, 4, 4, 154880), (16, 4, 4, 248320)]

# Profiling selection remains independent of pytest's case selection.
perf_shape = [
    (1, 4, 4, 154880),
    (32, 4, 4, 154880),
    (1, 4, 4, 248320),
    (16, 4, 4, 248320),
]


def generate_test_data(shape, case):
    """Keep the original CPU fixtures and random seeds for every parameter row."""
    batch_size, num_draft_tokens, _, vocab_size = shape
    with torch.device("cpu"):
        if case == "general-tree":
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
        elif case == "thresholds":
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
                    target_probs[req_idx, step - 1] /= target_probs[
                        req_idx, step - 1
                    ].sum()

            uniforms = torch.tensor([[0.1, 0.9, 0.2, 0.0], [0.7, 0.2, 0.95, 0.0]])
            final_uniforms = torch.tensor([0.25, 0.75])
            retrive_index = torch.arange(
                batch_size * num_draft_tokens, dtype=torch.int64
            ).view(batch_size, num_draft_tokens)
            retrive_next_token = torch.arange(
                1, num_draft_tokens + 1, dtype=torch.int64
            ).repeat(batch_size, 1)
            retrive_next_token[:, -1] = -1
            retrive_next_sibling = torch.full_like(retrive_next_token, -1)
        else:
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
                    target_probs[req_idx, step - 1] /= target_probs[
                        req_idx, step - 1
                    ].sum()

            uniforms = torch.rand(batch_size, num_draft_tokens)
            # A random coin can land within FP32 reduction error of a CDF boundary for
            # large vocabularies. Use the midpoint of a high-mass token's interval so
            # exact token equality tests the algorithm instead of reduction order.
            batch_size, num_draft_tokens = candidates.shape
            final_uniforms = torch.empty(batch_size, dtype=torch.float32)

            for req_idx in range(batch_size):
                num_accepted = 0
                rejected_token = -1
                for step in range(1, num_draft_tokens):
                    draft_token = int(candidates[req_idx, step])
                    target_prob = float(target_probs[req_idx, step - 1, draft_token])
                    if float(uniforms[req_idx, step - 1]) <= target_prob:
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
            retrive_index = torch.arange(
                batch_size * num_draft_tokens, dtype=torch.int64
            ).view(batch_size, num_draft_tokens)
            retrive_next_token = torch.arange(
                1, num_draft_tokens + 1, dtype=torch.int64
            ).repeat(batch_size, 1)
            retrive_next_token[:, -1] = -1
            retrive_next_sibling = torch.full_like(retrive_next_token, -1)
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


@pytest.mark.parametrize(
    "shape,case,threshold_single,threshold_acc",
    [
        pytest.param(
            shape,
            "chain",
            1.0,
            1.0,
            id=model,
            marks=() if shape in default_shapes else pytest.mark.extra_checks,
        )
        for model, shapes in model_cases.items()
        for shape in shapes
    ]
    + [
        pytest.param(
            (batch, drafts, drafts, vocab),
            "chain",
            1.0,
            1.0,
            id=f"legacy-chain-b{batch}-d{drafts}-v{vocab}",
            marks=pytest.mark.extra_checks,
        )
        for batch in (1, 4, 17)
        for drafts in (2, 5)
        for vocab in (20, 32000, 151552)
    ]
    + [
        pytest.param(
            shape,
            case,
            threshold_single,
            threshold_acc,
            id=f"{case}-{threshold_single}-{threshold_acc}",
            marks=(
                ()
                if case == "general-tree" and threshold_single == 1.0
                else pytest.mark.extra_checks
            ),
        )
        for shape, case in [
            ((2, 6, 4, 20), "general-tree"),
            ((2, 4, 4, 32), "thresholds"),
        ]
        for threshold_single, threshold_acc in [(1.0, 1.0), (0.0, 0.0), (0.5, 0.8)]
    ],
)
def test_tree_speculative_sampling_target_only(
    shape, case, threshold_single, threshold_acc
):
    from sgl_kernel_npu.operators.speculative import (
        tree_speculative_sampling_target_only,
    )

    inputs = generate_test_data(shape, case)
    batch, candidates, max_tree_depth, _ = shape
    outputs = {
        "predicts": torch.full((batch * candidates,), -1, dtype=torch.int32),
        "accept_index": torch.full((batch, max_tree_depth), -1, dtype=torch.int32),
        "accept_token_num": torch.zeros(batch, dtype=torch.int32),
        "draft_probs": torch.empty_like(inputs["target_probs"]),
    }
    options = {
        "threshold_single": threshold_single,
        "threshold_acc": threshold_acc,
        "deterministic": True,
    }
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(tree_speculative_sampling_target_only, impl)
    else:
        require_distinct_impls(tree_speculative_sampling_target_only, impl, impl_ref)
        snapshots = {name: value.clone() for name, value in inputs.items()}
        reference = {name: value.clone() for name, value in outputs.items()}
        for backend, buffers in ((impl_ref, reference), (impl, outputs)):
            result = tree_speculative_sampling_target_only(
                **buffers, **inputs, **options, impl=backend
            )
            assert result is None
            for name, value in inputs.items():
                assert torch.equal(value, snapshots[name]), f"{backend} modified {name}"
        assert_accuracy(outputs, reference, standard=SCRATCH_STANDARD)
        del snapshots, reference

    executor = make_executor(
        tree_speculative_sampling_target_only,
        kwargs={**outputs, **inputs, **options},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=shape in perf_shape)
