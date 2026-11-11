"""MLA preprocess accuracy through the public attention interface.

The parameter combinations reproduce the legacy test_mla_preprocess cases; the
additional cases cover the combined cache mode, unaligned token counts and the
minimum supported shapes. The independent reference is the CPU PyTorch path
from the legacy test (golden2_pytorch), extended with the physical cache writes.
"""

import numpy as np
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

# INT8 values may differ by one code when an FP32 reduction order changes a
# rounding boundary; FP16/BF16 use the shared elementwise budgets.
_INT8_STANDARD = {"torch.int8": AccuracyStandard(atol=1, rtol=0)}

HIDDEN = 7168
MM1_OUT = 2112
Q_RMS = 1536
K_NOPE = 512
K_PE = 64
HEADS_Q = 16
Q_NOPE_DIM = 128
Q_DIM = 192
BLOCK_NUM = 1078
BLOCK_SIZE = 128

# Legacy combinations (N, headNum, hiddenDim) plus unaligned and minimum shapes.
LEGACY_COMBINATIONS = [
    (1, 32, 7168),
    (1, 64, 7168),
    (1, 128, 7168),
    (16, 32, 7168),
    (16, 64, 7168),
    (16, 128, 7168),
    (31, 32, 7168),
    (31, 64, 7168),
    (31, 128, 7168),
    (31, 128, 6144),
]
EXTRA_COMBINATIONS = [
    (2, 64, 7168),
    (7, 64, 7168),
    (64, 64, 7168),
    (127, 32, 7168),
    (1, 32, 6144),
]

# Production (N, heads, hidden, cache_mode, dtype) selections.
perf_shape = [
    (1, 32, 7168, "krope_ctkv", torch.bfloat16),
    (16, 64, 7168, "krope_ctkv", torch.bfloat16),
    (31, 128, 6144, "krope_ctkv", torch.bfloat16),
]


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat: torch.Tensor, block_size=(16, 16)) -> torch.Tensor:
    """ND -> NZ, preserving the legacy weight preparation."""
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    nd_mat = torch.nn.functional.pad(
        nd_mat, ((0, r - nd_mat.shape[0], 0, c - nd_mat.shape[1]))
    )
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat, (1, nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3])
    )
    return nz_mat.contiguous()


def trans_descale_param(fp32_deq_scale):
    """Legacy FP16 scale packing: truncate the mantissa and set bit 46."""
    uint32__deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32__deq_scale &= 0xFFFFE000
    fp32_deq_scale = np.frombuffer(uint32__deq_scale, np.float32)
    uint64_deq_scale = np.zeros(fp32_deq_scale.shape, np.uint64)
    uint64_deq_scale |= np.uint64(uint32__deq_scale)
    uint64_deq_scale |= 1 << 46
    return torch.from_numpy(np.int64(uint64_deq_scale))


def generate_data(tokens, head_num, hidden_dim, dtype, cache_mode):
    """Create NZ weights, aligned scales and physical caches for one case."""
    import torch_npu

    generator = torch.Generator(device="cpu").manual_seed(42)

    def uniform(low, high, shape):
        return (
            torch.rand(shape, generator=generator, device="cpu") * (high - low) + low
        ).to(dtype)

    hidden = uniform(-2.0, 2.0, (tokens, hidden_dim)).npu()
    gamma0 = torch.ones([hidden_dim], dtype=dtype, device="cpu").npu()
    beta0 = torch.zeros([hidden_dim], dtype=dtype, device="cpu").npu()

    quant_scale0 = uniform(-2.0, 2.0, (1,)).npu()
    quant_offset0 = (
        torch.randint(
            -128, 127, (1,), generator=generator, dtype=torch.int32, device="cpu"
        )
        .to(torch.int8)
        .npu()
    )

    wdqkv_nd = (
        (torch.rand(MM1_OUT, hidden_dim, generator=generator, device="cpu") * 4 - 2)
        .to(torch.int8)
        .npu()
    )
    bias0 = torch.randint(
        -10, 10, (MM1_OUT,), generator=generator, dtype=torch.int32, device="cpu"
    ).npu()
    descale0 = (
        (torch.rand(MM1_OUT, generator=generator, device="cpu") / 1000).float().npu()
    )

    gamma1 = uniform(-1.0, 1.0, (Q_RMS,)).npu()
    beta1 = (
        torch.randint(
            -2, 2, (Q_RMS,), generator=generator, dtype=torch.int32, device="cpu"
        )
        .to(dtype)
        .npu()
    )
    gamma2 = uniform(-1.0, 1.0, (K_NOPE,)).npu()

    quant_scale1 = uniform(-2.0, 2.0, (1,)).npu()
    quant_offset1 = (
        torch.randint(
            -128, 127, (1,), generator=generator, dtype=torch.int32, device="cpu"
        )
        .to(torch.int8)
        .npu()
    )

    wuq_nd = (
        (torch.rand(head_num * Q_DIM, Q_RMS, generator=generator, device="cpu") * 4 - 2)
        .to(torch.int8)
        .npu()
    )
    bias1 = torch.randint(
        -10,
        10,
        (head_num * Q_DIM,),
        generator=generator,
        dtype=torch.int32,
        device="cpu",
    ).npu()
    descale1 = (
        (torch.rand(head_num * Q_DIM, generator=generator, device="cpu") / 1000)
        .float()
        .npu()
    )

    wuk = uniform(-2.0, 2.0, (head_num, Q_NOPE_DIM, K_NOPE)).npu()

    cos = uniform(-1.0, 1.0, (tokens, K_PE)).npu()
    sin = uniform(-1.0, 1.0, (tokens, K_PE)).npu()

    slotmapping = (
        torch.randperm(BLOCK_NUM * BLOCK_SIZE, generator=generator, device="cpu")[
            :tokens
        ]
        .to(torch.int32)
        .npu()
    )

    int8_cache = cache_mode == "int8_nzcache"
    nope_dim = 576 if cache_mode == "kvcache" else K_NOPE
    kv_cache = torch.zeros(
        BLOCK_NUM, BLOCK_SIZE, 1, nope_dim, dtype=torch.int8 if int8_cache else dtype
    ).npu()
    kv_cache_rope = torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, K_PE, dtype=dtype).npu()
    q_out0 = torch.zeros(
        tokens, head_num, nope_dim, dtype=torch.int8 if int8_cache else dtype
    ).npu()
    q_out1 = torch.zeros(tokens, head_num, K_PE, dtype=dtype).npu()

    ctkv_scale = uniform(-2.0, 2.0, (1,)).npu()
    q_nope_scale = uniform(-1.0, 1.0, (head_num,)).npu()

    if dtype == torch.float16:
        descale0 = trans_descale_param(np.array(descale0.cpu())).npu()
        descale1 = trans_descale_param(np.array(descale1.cpu())).npu()

    # Keep logical ND weights for the supplementary torch_npu golden.
    wdqkv = transdata(wdqkv_nd, (16, 32))
    wdqkv = torch_npu.npu_format_cast(wdqkv, 29).contiguous()
    wuq = transdata(wuq_nd, (16, 32))
    wuq = torch_npu.npu_format_cast(wuq, 29).contiguous()

    data = dict(
        hidden_state=hidden,
        gamma0=gamma0,
        beta0=beta0,
        wdqkv=wdqkv,
        descale0=descale0,
        gamma1=gamma1,
        beta1=beta1,
        wuq=wuq,
        descale1=descale1,
        gamma2=gamma2,
        cos=cos,
        sin=sin,
        wuk=wuk,
        kv_cache=kv_cache,
        kv_cache_rope=kv_cache_rope,
        slotmapping=slotmapping,
        quant_scale0=quant_scale0,
        quant_offset0=quant_offset0,
        bias0=bias0,
        quant_scale1=quant_scale1,
        quant_offset1=quant_offset1,
        bias1=bias1,
        ctkv_scale=ctkv_scale,
        q_nope_scale=q_nope_scale,
        cache_mode=cache_mode,
        quant_mode="per_tensor_quant_asymm",
        q_out0=q_out0,
        kv_cache_out0=kv_cache,
        q_out1=q_out1,
        kv_cache_out1=kv_cache_rope,
    )
    return data, wdqkv_nd, wuq_nd


_MUTABLE = ("kv_cache", "kv_cache_rope", "q_out0", "q_out1")


def clone_inputs(data):
    """Clone every mutated tensor; ``kv_cache_out0`` keeps aliasing kv_cache."""
    result = dict(data)
    for name in _MUTABLE:
        result[name] = data[name].clone()
    result["kv_cache_out0"] = result["kv_cache"]
    result["kv_cache_out1"] = result["kv_cache_rope"]
    return result


def decode_cache(cache, cache_mode, name):
    """Expose logical [slots, dim] rows on CPU for ND and group-major NZ layouts."""
    cache = cache.cpu()
    if cache_mode in ("kvcache", "krope_ctkv"):
        return cache.reshape(-1, cache.shape[-1])
    assert name in ("kv_cache", "kv_cache_rope")
    c0 = 32 if (cache_mode == "int8_nzcache" and name == "kv_cache") else 16
    groups = cache.shape[-1] // c0
    return (
        cache.view(cache.shape[0], groups, BLOCK_SIZE, c0)
        .permute(0, 2, 1, 3)
        .reshape(-1, cache.shape[-1])
    )


def check_result(actual, expected, actual_inputs, expected_inputs, before, data):
    slots = data["slotmapping"].long().cpu()
    assert_accuracy(actual["q_out0"], expected["q_out0"], _INT8_STANDARD, path="q_out0")
    assert_accuracy(actual["q_out1"], expected["q_out1"], _INT8_STANDARD, path="q_out1")
    for name in ("kv_cache", "kv_cache_rope"):
        actual_rows = decode_cache(actual_inputs[name], data["cache_mode"], name)
        expected_rows = decode_cache(expected_inputs[name], data["cache_mode"], name)
        before_rows = decode_cache(before[name], data["cache_mode"], name)
        untouched = torch.ones(actual_rows.shape[0], dtype=torch.bool, device="cpu")
        untouched[slots] = False
        assert torch.equal(
            actual_rows[untouched], before_rows[untouched]
        ), f"{name}: unwritten slot changed"
        assert_accuracy(
            actual_rows[slots], expected_rows[slots], _INT8_STANDARD, path=name
        )


def run_case(tokens, head_num, hidden_dim, dtype, cache_mode, *, enable_perf=False):
    from sgl_kernel_npu.operators.attention import mla_preprocess

    data, _, _ = generate_data(tokens, head_num, hidden_dim, dtype, cache_mode)
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    before = clone_inputs(data)

    if skip_accuracy:
        require_test_impl(mla_preprocess, impl)
        accuracy_checked = False
    else:
        require_distinct_impls(mla_preprocess, impl, impl_ref)
        reference_inputs = clone_inputs(data)
        actual_inputs = clone_inputs(data)
        expected = mla_preprocess(**reference_inputs, impl=impl_ref)
        actual = mla_preprocess(**actual_inputs, impl=impl)
        expected = dict(
            zip(("q_out0", "kv_cache_out0", "q_out1", "kv_cache_out1"), expected)
        )
        actual = dict(
            zip(("q_out0", "kv_cache_out0", "q_out1", "kv_cache_out1"), actual)
        )
        check_result(actual, expected, actual_inputs, reference_inputs, before, data)
        accuracy_checked = True

    executor = make_executor(
        mla_preprocess,
        kwargs=clone_inputs(data),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=accuracy_checked,
    )
    check_performance(
        executor,
        enable_perf=enable_perf
        and (tokens, head_num, hidden_dim, cache_mode, dtype) in perf_shape,
    )


@pytest.fixture
def enable_internal_format():
    import torch_npu

    # torch_npu exposes allow_internal_format as a write-only option: reading it
    # back raises AttributeError, so the legacy read-modify-restore pattern does
    # not work. The legacy test also enables it for the whole process.
    torch_npu.npu.config.allow_internal_format = True
    yield


@pytest.mark.usefixtures("enable_internal_format")
@pytest.mark.parametrize(
    "cache_mode,dtype",
    [
        ("krope_ctkv", torch.bfloat16),
        ("int8_nzcache", torch.bfloat16),
        ("nzcache", torch.bfloat16),
        ("krope_ctkv", torch.float16),
    ],
)
@pytest.mark.parametrize(
    "shape", LEGACY_COMBINATIONS, ids=lambda s: f"n{s[0]}-h{s[1]}-d{s[2]}"
)
def test_mla_preprocess_legacy_cases(shape, cache_mode, dtype):
    """Legacy N/head/hidden combinations and cache mode/dtype coverage."""
    run_case(*shape, dtype=dtype, cache_mode=cache_mode, enable_perf=True)


@pytest.mark.usefixtures("enable_internal_format")
@pytest.mark.parametrize(
    "cache_mode", ["kvcache", "krope_ctkv", "int8_nzcache", "nzcache"]
)
@pytest.mark.parametrize(
    "dtype,cache_modes",
    [
        (torch.bfloat16, ["kvcache", "krope_ctkv", "int8_nzcache", "nzcache"]),
        (torch.float16, ["kvcache", "krope_ctkv"]),
    ],
)
@pytest.mark.parametrize(
    "shape", EXTRA_COMBINATIONS, ids=lambda s: f"n{s[0]}-h{s[1]}-d{s[2]}"
)
def test_mla_preprocess_extended(shape, cache_mode, dtype, cache_modes):
    """Unaligned token counts, minimum shapes and the combined 576-wide cache."""
    if cache_mode not in cache_modes:
        pytest.skip("float16 supports cache_mode 0 and 1 only")
    run_case(*shape, dtype=dtype, cache_mode=cache_mode)


def _golden_torch_npu(data, wdqkv_nd, wuq_nd, head_num, dtype):
    """Legacy golden1: torch_npu small ops on logical ND weights."""
    import torch_npu

    def quant_per_tensor(x, scale, zp):
        x = x / scale.float() + zp.float()
        x = x.to(torch.float16)
        return torch.round(torch.clamp(x, -128, 127)).to(torch.int8)

    def apply_rope(x, cos, sin):
        x = x.float()
        cos = cos.float().unsqueeze(1)
        sin = sin.float().unsqueeze(1)
        first, second = torch.chunk(x, 2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        return (x * cos + rotated * sin).to(dtype)

    quant0 = quant_per_tensor(
        data["hidden_state"], data["quant_scale0"], data["quant_offset0"]
    )
    fused = torch_npu.npu_quant_matmul(
        quant0,
        wdqkv_nd.transpose(0, 1),
        data["descale0"],
        bias=data["bias0"],
        output_dtype=dtype,
    )
    latent, q = fused.split([K_NOPE + K_PE, Q_RMS], dim=-1)
    k_nope = latent[..., :K_NOPE]
    k_pe = latent[..., K_NOPE:].unsqueeze(1)

    q = (
        torch_npu.npu_rms_norm(q.float(), data["gamma1"].float(), 1e-6)[0]
        + data["beta1"].float()
    )
    k_nope = torch_npu.npu_rms_norm(k_nope.float(), data["gamma2"].float(), 1e-6)[
        0
    ].unsqueeze(1)

    quant1 = quant_per_tensor(q, data["quant_scale1"], data["quant_offset1"])
    q_full = torch_npu.npu_quant_matmul(
        quant1,
        wuq_nd.transpose(0, 1),
        data["descale1"],
        bias=data["bias1"],
        output_dtype=dtype,
    ).view(-1, head_num, Q_DIM)
    q_nope, q_pe = q_full.split([Q_NOPE_DIM, Q_DIM - Q_NOPE_DIM], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), data["wuk"]).transpose(0, 1)
    q_pe_rope = apply_rope(q_pe, data["cos"], data["sin"])
    k_pe_rope = apply_rope(k_pe, data["cos"], data["sin"])

    if data["cache_mode"] == "int8_nzcache":
        q_nope_out = torch.round(
            torch.clamp(
                (
                    q_nope_out.float()
                    * data["q_nope_scale"].float().reshape(1, head_num, 1)
                ).to(torch.float16),
                -128,
                127,
            )
        ).to(torch.int8)
        k_nope = torch.round(
            torch.clamp(
                (k_nope.float() / data["ctkv_scale"].float()).to(torch.float16),
                -128,
                127,
            )
        ).to(torch.int8)
    else:
        q_nope_out = q_nope_out.to(dtype)
        k_nope = k_nope.to(dtype)
    return q_nope_out, q_pe_rope, k_nope.squeeze(1), k_pe_rope.squeeze(1)


@pytest.mark.extra_checks
@pytest.mark.usefixtures("enable_internal_format")
@pytest.mark.parametrize(
    "cache_mode,dtype",
    [
        ("krope_ctkv", torch.bfloat16),
        ("int8_nzcache", torch.bfloat16),
        ("nzcache", torch.bfloat16),
        ("krope_ctkv", torch.float16),
    ],
)
def test_mla_preprocess_npu_small_ops_reference(cache_mode, dtype):
    """Supplementary legacy golden1 parity on the production shape."""
    from sgl_kernel_npu.operators.attention import mla_preprocess

    impl, _ = get_test_impls()
    require_test_impl(mla_preprocess, impl)
    data, wdqkv_nd, wuq_nd = generate_data(1, HEADS_Q, HIDDEN, dtype, cache_mode)
    pointwise = _golden_torch_npu(data, wdqkv_nd, wuq_nd, HEADS_Q, dtype)

    actual_inputs = clone_inputs(data)
    actual = mla_preprocess(**actual_inputs, impl=impl)
    actual = dict(zip(("q_out0", "kv_cache_out0", "q_out1", "kv_cache_out1"), actual))
    q_nope, q_pe, k_nope, k_pe = pointwise
    assert_accuracy(actual["q_out0"], q_nope, _INT8_STANDARD, path="q_nope")
    assert_accuracy(actual["q_out1"], q_pe, _INT8_STANDARD, path="q_pe")
    slots = data["slotmapping"].long().cpu()
    assert_accuracy(
        decode_cache(actual_inputs["kv_cache"], cache_mode, "kv_cache")[slots],
        k_nope.cpu(),
        _INT8_STANDARD,
        path="k_nope",
    )
    assert_accuracy(
        decode_cache(actual_inputs["kv_cache_rope"], cache_mode, "kv_cache_rope")[
            slots
        ],
        k_pe.cpu(),
        _INT8_STANDARD,
        path="k_pe",
    )
