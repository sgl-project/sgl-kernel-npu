import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_skip_acc_check
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Source: sglang/srt/configs/glm5_next.py: hidden_size=4096, hc_mult=4.
# GLM-5.3-Flash cases borrow these dimensions; token counts are illustrative,
# not traced production inputs. Small-width cases are synthetic edge tests.
perf_shape = [(1, 4096), (128, 4096)]

# Avoid the shared BF16 atol=0.05 masking errors in inputs scaled by 0.1.
POST_STANDARD = {
    "torch.float16": AccuracyStandard(atol=1e-5, rtol=2e-3),
    "torch.bfloat16": AccuracyStandard(atol=1e-5, rtol=1e-2),
}


def generate_data(tokens, hidden, dtype, mode):
    torch.manual_seed(43)
    x = torch.randn(tokens, 4, hidden, dtype=dtype) * 0.1
    h_out = torch.randn(tokens, hidden, dtype=dtype) * 0.1
    h_post = torch.rand(tokens, 4, dtype=torch.float32)
    # Deliberately asymmetric: catches a transposed residual mapping.
    h_res = torch.rand(tokens, 4, 4, dtype=torch.float32).softmax(-1)
    # Mimic pre's doubly stochastic residual map without calling pre.
    for _ in range(20):
        h_res = h_res / h_res.sum(-2, keepdim=True)
        h_res = h_res / h_res.sum(-1, keepdim=True)
    if mode == "permutation":
        # Columns are output streams: this asymmetric permutation gives
        # output streams [3,0,1,2], and catches a transposed reference.
        h_res = torch.eye(4, dtype=torch.float32)[[1, 2, 3, 0]].repeat(tokens, 1, 1)
        x = torch.tensor([1, 2, 4, 8], dtype=dtype)[None, :, None].repeat(
            tokens, 1, hidden
        )
        h_post.zero_()
    elif mode == "identity":
        h_res = torch.eye(4, dtype=torch.float32).repeat(tokens, 1, 1)
        h_post.zero_()
    elif mode == "post_only":
        h_res.zero_()
    return x, h_res, h_out, h_post


@pytest.mark.parametrize(
    "tokens,hidden,mode",
    [
        pytest.param(1, 128, "normal", id="edge-minimum-1x4x128"),
        pytest.param(2, 128, "normal", id="edge-two-rows-2x4x128"),
        pytest.param(5, 256, "normal", id="edge-odd-rows-5x4x256"),
        pytest.param(3, 128, "permutation", id="edge-exact-permutation-3x4x128"),
        pytest.param(3, 128, "identity", id="edge-identity-3x4x128"),
        pytest.param(3, 128, "post_only", id="edge-post-only-3x4x128"),
        pytest.param(1, 4096, "normal", id="glm53-flash-decode-1x4x4096"),
        pytest.param(128, 4096, "normal", id="glm53-flash-extend-128x4x4096"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_post(tokens, hidden, mode, dtype):
    from sgl_kernel_npu.operators.hyper_conn import mhc_post

    inputs = generate_data(tokens, hidden, dtype, mode)
    executor = check_accuracy(mhc_post, args=inputs, standard=POST_STANDARD)
    if mode == "permutation" and get_skip_acc_check() is not True:
        # Literal stream selection is independent of the Torch bmm reference.
        actual = executor()
        assert torch.equal(actual, inputs[0][:, [3, 0, 1, 2], :])
    check_performance(
        executor,
        enable_perf=(tokens, hidden) in perf_shape and dtype == torch.bfloat16,
    )


@pytest.mark.parametrize("tokens", [1, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_pre_post_chain_matches_pr21(tokens, dtype):
    """Compare the migrated pre/post chain with PR #21's direct vendor calls.

    Uses synthetic tensors with GLM config dimensions, not captured activations
    or checkpoint weights; this is not a full-model accuracy validation.
    """
    from cann_ops_transformer.ops import mhc_post as vendor_post
    from cann_ops_transformer.ops import mhc_pre_sinkhorn as vendor_pre
    from sgl_kernel_npu.operators.hyper_conn import mhc_post, mhc_pre_sinkhorn

    torch.manual_seed(123)  # Held out from the earlier nine-seed calibration.
    x = torch.randn(tokens, 4, 4096, dtype=dtype) * 0.1
    phi = torch.randn(24, 16384, dtype=torch.float32) / 128
    alpha = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    bias = torch.randn(24, dtype=torch.float32) * 0.1
    layer_out = torch.randn(tokens, 4096, dtype=dtype) * 0.1

    baseline = vendor_pre(x.unsqueeze(1), phi, alpha, bias, 4, 20, 1e-6, 1e-6)
    baseline = tuple(t.squeeze(1) for t in baseline)
    actual = mhc_pre_sinkhorn(x, phi, alpha, bias, impl="cann_ops_transformer")
    for result, expected in zip(actual, baseline):
        assert result.shape == expected.shape and result.dtype == expected.dtype
        assert torch.equal(result, expected)
    expected_out = vendor_post(
        x, baseline[2].view(tokens, 4, 4), layer_out, baseline[1]
    )
    actual_out = mhc_post(
        x,
        actual[2].view(tokens, 4, 4),
        layer_out,
        actual[1],
        impl="cann_ops_transformer",
    )
    assert (
        actual_out.shape == expected_out.shape
        and actual_out.dtype == expected_out.dtype
    )
    assert torch.equal(actual_out, expected_out)
