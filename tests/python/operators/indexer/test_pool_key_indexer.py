import pytest
import torch

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

# W: synthetic one-token-per-request PA workload, using GLM-5.3-Flash's
# 32 index heads, 128 channels and pool=4; not a captured production tensor.
perf_shape = [(16, 512, 32)]


@pytest.fixture(scope="module", autouse=True)
def require_cann_9_2_0_or_newer():
    from packaging.version import Version
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version

    version = detect_cann_version()
    if Version(version) < Version("9.2.0"):
        pytest.skip(f"pool_key_indexer requires CANN >= 9.2.0 (found {version})")


def generate_test_data(
    layout, rows, n_pools, heads, dtype, topk=16, return_value=False
):
    """Positive, separated scores make pool ranking independent of roundoff."""
    pool_size = 4
    quant_mode = -1
    q_descale = None
    k_descale = None
    if dtype == torch.int8:
        quant_mode = 2
    elif dtype == torch.float8_e4m3fn:
        quant_mode = 0

    if quant_mode == -1:
        q = torch.zeros(rows, 1, heads, 128, dtype=dtype)
        q[..., 0] = 1
        weights = torch.ones(rows, 1, heads, dtype=dtype)
        keys = torch.zeros(rows, n_pools, 1, 128, dtype=dtype)
        keys[..., 0] = torch.arange(1, n_pools + 1, dtype=dtype).view(1, n_pools, 1)
    else:
        # Selecting every pool keeps the quantized production case independent
        # of tie ordering while still exercising descale forwarding.
        q = torch.ones(rows, 1, heads, 128, dtype=torch.float32).to(dtype)
        keys = torch.ones(rows, n_pools, 1, 128, dtype=torch.float32).to(dtype)
        weight_dtype = torch.float16 if quant_mode == 2 else torch.bfloat16
        scale_dtype = torch.float16 if quant_mode == 2 else torch.float32
        weights = torch.ones(rows, 1, heads, dtype=weight_dtype)
        q_descale = torch.ones(rows, 1, heads, dtype=scale_dtype)
        k_descale = torch.ones(rows, n_pools, 1, dtype=scale_dtype)
    inputs = dict(
        query=q,
        pool_key=keys,
        weights=weights,
        pool_tail_k=torch.full((rows,), 3, dtype=torch.int64),
        topk=topk,
        pool_size=pool_size,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        quant_mode=quant_mode,
        return_value=return_value,
        q_descale=q_descale,
        k_descale=k_descale,
    )
    if layout == "TND":
        inputs.update(
            query=q.squeeze(1),
            weights=weights.squeeze(1),
            q_descale=None if q_descale is None else q_descale.squeeze(1),
            layout_q="TND",
            layout_k="PA_BBND",
            actual_seq_q=torch.arange(1, rows + 1, dtype=torch.int64),
            actual_seq_k=torch.full((rows,), n_pools, dtype=torch.int64),
            block_table=torch.arange(rows, dtype=torch.int32).view(rows, 1),
        )
    elif layout != "BSND":
        raise ValueError(layout)
    return inputs


def canonicalize_selected_pools(result, topk, pool_size):
    """CANN does not guarantee a stable selected-pool order; tail order is fixed."""
    indices, values = result
    if values.numel():
        order = indices[..., :topk:pool_size].argsort(dim=-1)
        values = values.gather(-1, order)
    indices = torch.cat(
        (indices[..., :topk].sort().values, indices[..., topk:]), dim=-1
    )
    return indices, values


# Six representative tuples replace layout x dtype x shape expansion.
@pytest.mark.parametrize(
    "layout,dtype,rows,n_pools,heads,topk,return_value",
    [
        # Case 01: Typical TND BF16: 16 requests, 512 pools, 32 heads, topk2048; profile.
        pytest.param(
            "TND",
            torch.bfloat16,
            16,
            512,
            32,
            2048,
            False,
            id="W-glm53-flash-tnd-topk2048",
        ),
        # Case 02: Typical BSND BF16 with the same full-pool selection.
        pytest.param(
            "BSND",
            torch.bfloat16,
            16,
            512,
            32,
            2048,
            False,
            id="W-glm53-flash-bsnd-topk2048",
        ),
        # Case 03: Minimum TND FP16: one request/head and a ranked subset.
        pytest.param(
            "TND",
            torch.float16,
            1,
            16,
            1,
            16,
            True,
            id="edge-tnd-fp16-minimum-values",
        ),
        # Case 04: Minimum BSND FP16 with a ranked subset.
        pytest.param(
            "BSND",
            torch.float16,
            1,
            16,
            1,
            16,
            True,
            id="edge-bsnd-fp16-minimum-values",
        ),
        # Case 05: TND BF16: odd request count and a ranked subset.
        pytest.param(
            "TND",
            torch.bfloat16,
            3,
            32,
            4,
            16,
            False,
            id="edge-odd-requests-ranked-subset",
        ),
        # Case 06: BSND BF16: multiple requests/heads and a ranked subset.
        pytest.param(
            "BSND",
            torch.bfloat16,
            16,
            64,
            32,
            16,
            True,
            id="edge-multi-head-ranked-subset-values",
        ),
        # Case 07: INT8 production TND shape; also profile this case on A3.
        pytest.param(
            "TND",
            torch.int8,
            16,
            512,
            32,
            2048,
            False,
            id="W-glm53-flash-tnd-topk2048-int8",
        ),
        # Case 08: FP8 production TND shape; also profile this case on A5.
        pytest.param(
            "TND",
            torch.float8_e4m3fn,
            16,
            512,
            32,
            2048,
            False,
            id="W-glm53-flash-tnd-topk2048-fp8",
        ),
    ],
)
def test_pool_key_indexer(layout, dtype, rows, n_pools, heads, topk, return_value):
    from sgl_kernel_npu.operators.indexer import pool_key_indexer

    if dtype in (torch.int8, torch.float8_e4m3fn):
        from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

        generation = get_hardware().generation
        if dtype == torch.int8 and generation != Generation.ARCH22:
            pytest.skip("INT8 pooled-key indexer coverage targets Ascend A3")
        if dtype == torch.float8_e4m3fn and generation != Generation.ARCH35:
            pytest.skip("FP8 pooled-key indexer coverage targets Ascend A5")

    impl, impl_ref = get_test_impls()
    inputs = generate_test_data(layout, rows, n_pools, heads, dtype, topk, return_value)

    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(pool_key_indexer, impl, impl_ref)
        actual = pool_key_indexer(**inputs, impl=impl)
        expected = pool_key_indexer(**inputs, impl=impl_ref)
        assert_accuracy(
            canonicalize_selected_pools(actual, topk, inputs["pool_size"]),
            canonicalize_selected_pools(expected, topk, inputs["pool_size"]),
        )
    else:
        require_test_impl(pool_key_indexer, impl)

    executor = make_executor(
        pool_key_indexer,
        kwargs=inputs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )

    # Performance: profile the public entry without sorting/comparison.
    check_performance(
        executor,
        enable_perf=(rows, n_pools, heads) in perf_shape
        and layout == "TND"
        and dtype in (torch.bfloat16, torch.int8, torch.float8_e4m3fn),
    )


def test_pool_key_indexer_int8_paged():
    from sgl_kernel_npu.operators.indexer import pool_key_indexer
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH22:
        pytest.skip("INT8 pooled-key indexer coverage targets Ascend A3")

    impl, _ = get_test_impls()
    require_test_impl(pool_key_indexer, impl)
    inputs = generate_test_data("TND", 4, 16, 32, torch.float16)
    inputs["query"] = inputs["query"].to(torch.int8)
    inputs["pool_key"] = inputs["pool_key"].to(torch.int8)
    inputs["quant_mode"] = 2
    q_descale = torch.ones_like(inputs["weights"])
    k_descale = torch.ones_like(inputs["pool_key"][..., 0], dtype=torch.float16)
    k_descale[:, :4, :] = 32

    indices, values = pool_key_indexer(
        **inputs,
        q_descale=q_descale,
        k_descale=k_descale,
        impl=impl,
    )

    # The scaled first four pools outrank the remaining twelve for every request.
    expected = torch.cat((torch.arange(16), torch.arange(64, 67))).to(torch.int32)
    ranked_indices = canonicalize_selected_pools((indices, values), 16, 4)[0]
    assert_accuracy(ranked_indices, expected.expand_as(ranked_indices))
    assert values.numel() == 0


@pytest.fixture
def require_cann_9_2_0_arch35():
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("FP8 pool_key_indexer requires Ascend A5 (arch35)")
    if detect_cann_version() != "9.2.0":
        pytest.skip("FP8 pool_key_indexer requires CANN 9.2.0")


@pytest.mark.usefixtures("require_cann_9_2_0_arch35")
@pytest.mark.parametrize("layout", ["BSND", "TND", "PA_BBND"])
def test_pool_key_indexer_fp8(layout):
    from sgl_kernel_npu.operators.indexer import pool_key_indexer

    torch.manual_seed(19)
    impl, _ = get_test_impls()
    q = torch.randn(2, 2, 2, 128).to(torch.float8_e4m3fn)
    k = torch.randn(2, 4, 1, 128).to(torch.float8_e4m3fn)
    weights = torch.rand(2, 2, 2).bfloat16()
    qs = torch.rand(2, 2, 2) + 0.1
    ks = torch.tensor([0.1, 0.25, 2.0, 4.0, 4.0, 2.0, 0.25, 0.1]).reshape(2, 4, 1)
    dequant_q = q.float() * qs.unsqueeze(-1)
    dequant_k = k.float() * ks.unsqueeze(-1)
    scores = torch.einsum("bqhd,bkd->bqhk", dequant_q, dequant_k[:, :, 0])
    scores = (scores * (128**-0.5)).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    values, pools = scores.topk(2, dim=-1)
    expected = torch.full((2, 2, 11), -1, dtype=torch.int32, device=q.device)
    expected[:, :, :8] = (
        pools.unsqueeze(-1) * 4 + torch.arange(4, device=q.device)
    ).reshape(2, 2, 8)
    kwargs = dict(layout_q="BSND", layout_k="BSND")
    if layout != "BSND":
        q = q.reshape(4, 2, 128)
        qs = qs.reshape(4, 2)
        weights = weights.reshape(4, 2)
        kwargs = dict(
            layout_q="TND", layout_k=layout, actual_seq_q=torch.tensor([2, 4])
        )
        if layout == "TND":
            k = k.reshape(8, 1, 128)
            ks = ks.reshape(8, 1)
            kwargs["actual_seq_k"] = torch.tensor([4, 8])
        else:
            pages = torch.zeros(
                2, 16, 1, 128, dtype=torch.float8_e4m3fn, device=k.device
            )
            scales = torch.ones(2, 16, 1, device=k.device)
            pages[:, :4] = k
            scales[:, :4] = ks
            k, ks = pages, scales
            kwargs["block_table"] = torch.tensor(
                [[0], [1]], dtype=torch.int32, device=k.device
            )
            kwargs["actual_seq_k"] = torch.tensor([4, 4])
        expected = expected.reshape(4, 11)
        values = values.reshape(4, 2)
    indices, actual_values = pool_key_indexer(
        q,
        k,
        weights,
        torch.tensor([0, 0]),
        topk=8,
        pool_size=4,
        mask_mode=0,
        quant_mode=0,
        return_value=True,
        q_descale=qs,
        k_descale=ks,
        impl=impl,
        **kwargs,
    )
    assert_accuracy(indices, expected)
    assert_accuracy(
        actual_values, values, {"torch.float32": AccuracyStandard(atol=5e-2, rtol=5e-2)}
    )
