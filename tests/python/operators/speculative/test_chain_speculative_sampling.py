"""Model-shaped chain sampling and in-place outputs through the public API.

The probability tensors at this operator boundary are always FP32.
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

# Group shape ownership by model.
model_cases = {
    "glm-4-7-flash": [(batch, 4, 3, 154880) for batch in (1, 2, 4, 8, 16, 32)],
    "qwen3-5-9b": [(batch, 4, 3, 248320) for batch in (1, 2, 4, 8, 16)],
    "deepseek-v4-flash": [
        # Keep both verified candidate-width configurations under the same model.
        *[(batch, 6, 5, 129280) for batch in (1, 2, 4, 8)],
        *[(batch, 7, 6, 129280) for batch in (1, 2, 4, 6, 8, 10, 12)],
    ],
}

# One representative per vocabulary/candidate-width configuration; all other
# confirmed shapes remain available through extra_checks.
default_shapes = [
    (1, 4, 3, 154880),
    (16, 4, 3, 248320),
    (8, 6, 5, 129280),
    (12, 7, 6, 129280),
]

# Profiling selection remains independent of pytest's case selection.
perf_shape = [
    (1, 4, 3, 154880),
    (32, 4, 3, 154880),
    (1, 4, 3, 248320),
    (16, 4, 3, 248320),
    (1, 6, 5, 129280),
    (8, 6, 5, 129280),
    (1, 7, 6, 129280),
    (12, 7, 6, 129280),
]


def generate_test_data(shape, acceptance, case="model"):
    """Preserve model, legacy and boundary inputs; only their test runner is shared.

    Legacy inputs: tests/python/sgl_kernel_npu/test_chain_speculative_sampling.py
    at sgl-project/sgl-kernel-npu commit 15ea25766eb0f9a30951d7d127af8600998272ea.
    """
    if case.startswith("legacy"):
        large_vocab = case == "legacy-large"
        with torch.device("cpu"):
            batch_size, num_draft_tokens = 2, 4
            retrive_index = torch.arange(batch_size * num_draft_tokens).view(
                batch_size, num_draft_tokens
            )
            if large_vocab:
                vocab_size = 151552
                candidates = torch.tensor([[0, 11, 22, 33], [0, 44, 55, 66]])
                target_probs = torch.zeros(batch_size, num_draft_tokens, vocab_size)
                draft_probs = torch.zeros(batch_size, num_draft_tokens - 1, vocab_size)
                for row, token in enumerate(candidates[0, 1:]):
                    target_probs[0, row, token] = 0.8
                    target_probs[0, row, 0] = 0.2
                    draft_probs[0, row, token] = 0.5
                    draft_probs[0, row, 0] = 0.5
                target_probs[0, -1, 0] = 0.25
                target_probs[0, -1, -1] = 0.75
                target_probs[1, 0, candidates[1, 1]] = 0.1
                target_probs[1, 0, -1] = 0.9
                draft_probs[1, 0, candidates[1, 1]] = 0.9
                draft_probs[1, 0, -1] = 0.1
                for row, token in enumerate(candidates[1, 2:], start=1):
                    target_probs[1, row, token] = 1.0
                    draft_probs[1, row, token] = 1.0
                target_probs[1, -1, -1] = 1.0
                uniforms = torch.tensor([[0.1, 0.1, 0.1, 0.0], [0.9, 0.0, 0.0, 0.0]])
                final_uniforms = torch.tensor([0.5, 0.5])
            else:
                vocab_size = 11
                candidates = torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]])
                target_probs = torch.softmax(
                    torch.tensor(
                        [
                            [
                                [0.1, 0.2, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                [0.1, 0.2, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                [0.1, 0.2, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                [0.1, 0.2, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            ],
                            [
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 2.0, 0.1, 0.1, 0.1],
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.2, 0.1, 0.1, 0.1],
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1],
                                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1],
                            ],
                        ],
                        dtype=torch.float32,
                    ),
                    dim=-1,
                )
                draft_probs = torch.softmax(
                    torch.tensor(
                        [
                            [
                                [0.1] * vocab_size,
                                [0.1] * vocab_size,
                                [0.1] * vocab_size,
                            ],
                            [
                                [0.1] * vocab_size,
                                [0.1] * vocab_size,
                                [0.1] * vocab_size,
                            ],
                        ],
                        dtype=torch.float32,
                    ),
                    dim=-1,
                )
                draft_probs[1, 0, 5] = 0.9
                draft_probs[1, 0] /= draft_probs[1, 0].sum()
                uniforms = torch.tensor([[0.1, 0.1, 0.1, 0.0], [0.99, 0.0, 0.0, 0.0]])
                final_uniforms = torch.tensor([0.37, 0.61])
            inputs = {
                "candidates": candidates,
                "retrive_index": retrive_index,
                "retrive_next_token": torch.full_like(candidates, -1),
                "retrive_next_sibling": torch.full_like(candidates, -1),
                "uniform_samples": uniforms,
                "uniform_samples_for_final_sampling": final_uniforms,
                "target_probs": target_probs,
                "draft_probs": draft_probs,
            }
        return {name: value.npu() for name, value in inputs.items()}

    batch, draft_tokens, draft_rows, vocab = shape
    candidates = (
        torch.arange(batch * draft_tokens, dtype=torch.int64).reshape(
            batch, draft_tokens
        )
        % vocab
    )
    retrive_index = torch.arange(batch * draft_tokens, dtype=torch.int64).reshape(
        batch, draft_tokens
    )
    target_probs = torch.full(
        (batch, draft_tokens, vocab), 1.0 / vocab, dtype=torch.float32
    )
    draft_probs = torch.full(
        (batch, draft_rows, vocab), 0.5 / vocab, dtype=torch.float32
    )
    request_indices = torch.arange(batch)
    for step in range(draft_tokens - 1):
        draft_probs[request_indices, step, candidates[:, step + 1]] += 0.5
    # Extra, unused draft rows still contain valid probability distributions.
    draft_probs[:, draft_tokens - 1 :, :] = 1.0 / vocab
    uniform_samples = torch.full(
        (batch, draft_tokens), 0.5 / vocab, dtype=torch.float32
    )
    if acceptance == "reject-first":
        uniform_samples[:, 0] = 0.5
    elif acceptance == "reject-partial":
        uniform_samples[:, 1] = 0.5
    inputs = {
        "candidates": candidates,
        "retrive_index": retrive_index,
        "retrive_next_token": torch.full_like(candidates, -1),
        "retrive_next_sibling": torch.full_like(candidates, -1),
        "uniform_samples": uniform_samples,
        "uniform_samples_for_final_sampling": torch.full(
            (batch,), 0.37, dtype=torch.float32
        ),
        "target_probs": target_probs,
        "draft_probs": draft_probs,
    }
    if case in ("block-tail", "permuted-slots"):
        inputs["uniform_samples_for_final_sampling"].fill_(0.9999)
    if case == "permuted-slots":
        inputs["retrive_index"] = (
            inputs["retrive_index"].flatten().flip(0).reshape(shape[0], shape[1])
        )
    elif case == "strict-acceptance":
        inputs["target_probs"][0, 0] = torch.tensor([0.25, 0.25, 0.25, 0.25])
        inputs["draft_probs"][0, 0] = torch.tensor([0.25, 0.5, 0.25, 0.0])
        # candidate=1, p=.25, q=.5 and u=.5: strict '<' rejects equality.
        inputs["uniform_samples"][0, 0] = 0.5
    elif case == "cdf-plateau":
        inputs["target_probs"][0, 0] = torch.tensor([0.25, 0.0, 0.75, 0.0, 0.0])
        inputs["uniform_samples_for_final_sampling"][0] = 0.25
    return inputs


@pytest.mark.parametrize(
    "shape,acceptance,case",
    [
        pytest.param(
            shape,
            acceptance,
            "model",
            id=f"{model}-{acceptance}",
            marks=() if shape in default_shapes else pytest.mark.extra_checks,
        )
        for model, shapes in model_cases.items()
        for shape in shapes
        for acceptance in ("all-accepted", "reject-first", "reject-partial")
    ]
    + [
        pytest.param(
            shape, "all-accepted", case, id=case, marks=pytest.mark.extra_checks
        )
        for shape, case in [
            ((2, 4, 3, 11), "legacy-small"),
            ((2, 4, 3, 151552), "legacy-large"),
            ((0, 4, 3, 11), "empty-batch"),
            ((1, 1, 1, 1), "root-only-one-token-vocab"),
            ((2, 4, 4, 2049), "block-tail"),
            ((2, 4, 3, 4097), "permuted-slots"),
            ((1, 2, 1, 4), "strict-acceptance"),
            ((1, 1, 1, 5), "cdf-plateau"),
        ]
    ],
)
def test_chain_speculative_sampling(shape, acceptance, case):
    from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

    inputs = generate_test_data(shape, acceptance, case)
    candidates = inputs["candidates"]
    outputs = {
        "predicts": torch.full((candidates.numel(),), -1, dtype=torch.int32),
        "accept_index": torch.full_like(candidates, -1, dtype=torch.int32),
        "accept_token_num": torch.zeros(candidates.shape[0], dtype=torch.int32),
    }
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(chain_speculative_sampling, impl)
    else:
        require_distinct_impls(chain_speculative_sampling, impl, impl_ref)
        snapshots = {name: value.clone() for name, value in inputs.items()}
        reference = {name: value.clone() for name, value in outputs.items()}
        for backend, buffers in ((impl_ref, reference), (impl, outputs)):
            result = chain_speculative_sampling(**buffers, **inputs, impl=backend)
            assert result is None
            for name, value in inputs.items():
                assert torch.equal(value, snapshots[name]), f"{backend} modified {name}"
        assert_accuracy(outputs, reference)
        del snapshots, reference

    executor = make_executor(
        chain_speculative_sampling,
        kwargs={**outputs, **inputs},
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize("invalid_input", ["probability-dtype", "candidate-stride"])
def test_chain_speculative_sampling_invalid_inputs(invalid_input):
    from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

    require_test_impl(chain_speculative_sampling, "triton")
    inputs = generate_test_data((1, 4, 3, 11), "all-accepted")
    if invalid_input == "probability-dtype":
        inputs["target_probs"] = inputs["target_probs"].to(torch.float16)
        error, message = TypeError, "target_probs and draft_probs must be torch.float32"
    else:
        storage = torch.zeros((1, 8), dtype=torch.int64)
        storage[:, ::2] = inputs["candidates"]
        inputs["candidates"] = storage[:, ::2]
        error, message = ValueError, "all tensors must be contiguous"
    outputs = {
        "predicts": torch.full((4,), -1, dtype=torch.int32),
        "accept_index": torch.full((1, 4), -1, dtype=torch.int32),
        "accept_token_num": torch.zeros(1, dtype=torch.int32),
    }
    with pytest.raises(error, match=message):
        chain_speculative_sampling(**outputs, **inputs, impl="triton")
