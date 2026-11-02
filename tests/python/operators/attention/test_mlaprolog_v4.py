"""MLAProlog V4 with and without RoPE: C16 and static C8 cache contracts.

Dimensions follow moonshotai/Kimi-K3 config.json, with 96 Q heads sharded
over attention TP sizes 1/4/16. Tensor values are generated, not loaded
from a checkpoint. Weights and scales follow SGLang's MLAProlog input layouts.
RoPE cases use the same dimensions to cover V4's optional rotation path;
they do not describe Kimi K3's no-RoPE model behavior.
Source: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json
The independent torch_native implementation is an accuracy reference on CPU;
its transfers must not be reported as an NPU performance baseline.
"""

import math

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

page_size = 128
# Representative TP4/TP16 shapes, both RoPE modes; other cases check accuracy only.
perf_shape = [
    (tokens, heads, "PA_BSND", use_rope)
    for heads in (24, 6)
    for tokens in (1, 4, 16, 32, 64)
    for use_rope in (False, True)
]
cache_cases = [(0, False), (3, False), (3, True), (2, False)]
_BF16_STANDARD = {"torch.float32": AccuracyStandard(atol=5e-2, rtol=5e-2)}
_QUERY_SCALE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-7, rtol=1e-2)}
# Local INT8 quantization criterion: every element differs by at most one code.
# Compare in FP32 to avoid INT8 subtraction overflow; this is not the FP8 rule.
_INT8_QUERY_NORM_STANDARD = {"torch.float32": AccuracyStandard(atol=1, rtol=0)}


def accuracy_cases():
    # Cover each branch with representative shapes instead of a full product.
    scenarios = (
        [
            ("perf-shape", heads, tokens, "PA_BSND", False)
            for heads in (24, 6)
            for tokens in (1, 4, 16, 32, 64)
        ]
        + [("nz-layout", 6, tokens, "PA_NZ", False) for tokens in (1, 16, 64)]
        + [("head-count", 96, 4, "PA_BSND", False)]
        + [("norm-disabled", 6, 17, "PA_BSND", False)]
        + [("padding", 6, 5, layout, True) for layout in ("PA_BSND", "PA_NZ")]
        + [
            ("token-length", 6, tokens, "PA_BSND", False)
            for tokens in (2, 3, 7, 8, 12, 24, 31, 40, 48, 56, 63)
        ]
    )
    for group, heads, tokens, layout, padding in scenarios:
        modes = [(0, False)] if group == "token-length" else cache_cases
        rope_modes = (False,) if group == "token-length" else (False, True)
        for mode, c8 in modes:
            if group == "norm-disabled" and c8:
                continue  # C8 already disables the normalized-query output.
            for use_rope in rope_modes:
                yield pytest.param(
                    # hidden, query rank, heads, head dim, KV rank, R dim.
                    (7168, 1536, heads, 128, 512, 64),
                    tokens,
                    layout,
                    padding,
                    mode,
                    c8,
                    use_rope,
                    id=(
                        f"{group}-kimi-k3-tp{96 // heads}-t{tokens}-{layout}"
                        f"-w{mode}-c{8 if c8 else 16}"
                        f"-{'rope' if use_rope else 'no-rope'}"
                    ),
                )


@pytest.fixture
def enable_arch22_internal_format():
    import torch_npu
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH22:
        yield
        return

    original = torch_npu.npu.config.allow_internal_format
    try:
        # Keep NZ weights enabled through accuracy checks and profiling.
        torch_npu.npu.config.allow_internal_format = True
        yield
    finally:
        torch_npu.npu.config.allow_internal_format = original


def generate_data(
    shape,
    tokens,
    weight_quant_mode,
    c8,
    cache_mode,
    query_norm_flag,
    *,
    padding=False,
    use_rope=False,
):
    """Create real NZ weights and distinct, nonconsecutive paged cache slots."""
    import torch_npu
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    generation = get_hardware().generation
    if weight_quant_mode == 3 and generation == Generation.ARCH22:
        pytest.skip("MLAProlog MXFP8 projections require arch35")
    if weight_quant_mode == 2 and generation != Generation.ARCH22:
        pytest.skip("Dynamic INT8 coverage targets arch22")

    hidden, rank, heads, head_dim, latent, r_dim = shape
    generator = torch.Generator(device="cpu").manual_seed(42)

    def random_bf16(*dims, scale=1.0):
        value = torch.randn(dims, generator=generator, device="cpu") * scale
        return value.to(torch.bfloat16).npu()

    slots = ([127, 128, 0, 383] + list(range(256, 256 + tokens)))[:tokens]
    if padding:
        # CANN ScatterCache ignores negative slots; Q is still computed.
        slots[-1] = -1
    inputs = dict(
        token_x=random_bf16(tokens, hidden),
        weight_dq=random_bf16(hidden, rank, scale=hidden**-0.5),
        weight_uq_qr=random_bf16(rank, heads * (head_dim + r_dim), scale=rank**-0.5),
        weight_uk=random_bf16(heads, head_dim, latent, scale=head_dim**-0.5),
        weight_dkv_kr=random_bf16(hidden, latent + r_dim, scale=hidden**-0.5),
        rmsnorm_gamma_cq=1 + random_bf16(rank, scale=0.05),
        rmsnorm_gamma_ckv=1 + random_bf16(latent, scale=0.05),
        kv_cache=torch.full((3, page_size, 1, latent), 0.125).to(
            torch.float8_e4m3fn if c8 else torch.bfloat16
        ),
        kr_cache=torch.full((3, page_size, 1, r_dim), 0.125, dtype=torch.bfloat16),
        # Exercise page edges, slot zero, the final slot, and unordered writes.
        cache_index=torch.tensor(slots, dtype=torch.int64),
        rope_sin=None,
        rope_cos=None,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=1 if c8 else 0,
        query_quant_mode=1 if c8 else 0,
    )
    if c8:
        inputs["quant_scale_ckv"] = torch.tensor([32.0], dtype=torch.float32)
    if weight_quant_mode in (2, 3):
        for name, scale_name in (
            ("token_x", "dequant_scale_x"),
            ("weight_dq", "dequant_scale_w_dq"),
            ("weight_uq_qr", "dequant_scale_w_uq_qr"),
            ("weight_dkv_kr", "dequant_scale_w_dkv_kr"),
        ):
            is_weight = name != "token_x"
            value = inputs[name].T.contiguous() if is_weight else inputs[name]
            if weight_quant_mode == 2:
                quantized, scales = torch_npu.npu_dynamic_quant(value)
                scales = scales.float().reshape(-1, 1)
                inputs[scale_name] = scales.T.contiguous() if is_weight else scales
            else:
                quantized, scales = torch_npu.npu_dynamic_mx_quant(
                    value, axis=-1, dst_type=torch.float8_e4m3fn, block_size=32
                )
                inputs[scale_name] = (
                    scales.contiguous()
                    .reshape(value.shape[0], -1)
                    .view(torch.float8_e8m0fnu)
                )
            inputs[name] = quantized.T.contiguous() if is_weight else quantized
    for name in ("weight_dq", "weight_uq_qr", "weight_dkv_kr"):
        inputs[name] = torch_npu.npu_format_cast(inputs[name], 29)
    if use_rope:
        # Nonzero, nonconsecutive positions exercise rotation even when T=1.
        positions = 17 + 3 * torch.arange(tokens, dtype=torch.float32, device="cpu")
        inverse_freq = 10000.0 ** (
            -torch.arange(0, r_dim, 2, dtype=torch.float32, device="cpu") / r_dim
        )
        angles = positions[:, None] * inverse_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        inputs["rope_sin"] = angles.sin().to(torch.bfloat16).npu()
        inputs["rope_cos"] = angles.cos().to(torch.bfloat16).npu()
    return inputs


def _independent_cache(inputs):
    return {
        **inputs,
        "kv_cache": inputs["kv_cache"].clone(),
        "kr_cache": inputs["kr_cache"].clone(),
    }


def _cache_rows(cache, cache_mode):
    """Decode the physical PA_NZ storage for slot-level assertions on CPU."""
    cache = cache.cpu()
    blocks, page, _, dim = cache.shape
    if cache_mode == "PA_NZ":
        c0 = 32 // cache.element_size()
        cache = cache.reshape(blocks, dim // c0, page, c0).permute(0, 2, 1, 3)
    return cache.reshape(blocks * page, dim)


def _assert_mixed_accuracy(actual, reference, *, dtype, path):
    """Compare finite FP32 values using the specified output format's budget."""
    # CANN mixed tolerance: 99% close elements plus a value-dependent error cap.
    # https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
    mantissa_bits, min_normal_exponent, error_floor = {
        torch.float8_e4m3fn: (3, -6, 1.0),
        torch.bfloat16: (7, -126, 0.1),
    }[dtype]
    tolerance = 2.0**-mantissa_bits
    assert (
        torch.isfinite(actual).all() and torch.isfinite(reference).all()
    ), f"{path}: expected finite results"
    error = (actual - reference).abs()
    matched = int((error <= tolerance + tolerance * reference.abs()).sum())
    max_index = int(error.argmax())
    max_error = error.flatten()[max_index].item()
    reference_at_max = reference.flatten()[max_index].abs().item()
    # Clamp the exponent so zero/subnormals use the format's minimum spacing.
    exponent = math.floor(math.log2(max(reference_at_max, 2.0**min_normal_exponent)))
    ulp = 2.0 ** (exponent - mantissa_bits)
    error_limit = max(error_floor, 32 * ulp)
    assert matched * 100 >= error.numel() * 99 and max_error <= error_limit, (
        f"{path}: matched={matched}/{error.numel()} "
        f"({matched / error.numel():.6%}, required >=99%), "
        f"max_abs_error={max_error:.9g}, limit={error_limit:.9g}, "
        f"flat_index={max_index}, abs_reference={reference_at_max:.9g}"
    )


def _check_contract(outputs, inputs):
    assert isinstance(outputs, tuple) and len(outputs) == 5
    tokens = inputs["token_x"].shape[0]
    rank = inputs["weight_dq"].shape[-1]
    heads, _, latent = inputs["weight_uk"].shape
    r_dim = inputs["kr_cache"].shape[-1]
    c8 = inputs["kv_cache_quant_mode"] == 1
    mode3 = inputs["weight_quant_mode"] == 3
    norm = inputs["query_norm_flag"]
    norm_scale_shape = (0,)
    if norm and inputs["weight_quant_mode"] != 0:
        norm_scale_shape = (tokens, rank // 32 if mode3 else 1)
    expected = (
        ((tokens, heads, latent), torch.float8_e4m3fn if c8 else torch.bfloat16),
        ((tokens, heads, r_dim), torch.bfloat16),
        ((tokens, heads, 1) if c8 else (0,), torch.float32),
        ((tokens, rank) if norm else (0,), inputs["weight_uq_qr"].dtype),
        (
            norm_scale_shape,
            torch.float8_e8m0fnu if mode3 else torch.float32,
        ),
    )
    for output, (dims, dtype) in zip(outputs, expected):
        assert tuple(output.shape) == dims
        assert output.dtype == dtype
        assert output.device == inputs["token_x"].device


def _accuracy(outputs, expected, inputs, reference, before):
    _check_contract(outputs, inputs)
    _check_contract(expected, reference)
    query = outputs[0].cpu().float()
    query_ref = expected[0].cpu().float()
    if inputs["kv_cache_quant_mode"] == 1:
        # Check the public scale separately: a matching error in the FP8 value
        # and reciprocal scale must not disappear in the reconstruction check.
        assert_accuracy(
            outputs[2], expected[2], _QUERY_SCALE_STANDARD, path="query_scale"
        )
        # Normalize reconstructed Q to the reference scale before applying the
        # E4M3 error budget. The two raw FP8 tensors may use different scales.
        query *= outputs[2].cpu() / expected[2].cpu()
        _assert_mixed_accuracy(
            query, query_ref, dtype=torch.float8_e4m3fn, path="query"
        )
    else:
        assert_accuracy(query, query_ref, _BF16_STANDARD, path="query")
    query_rope = outputs[1].cpu().float()
    query_rope_ref = expected[1].cpu().float()
    if inputs["kv_cache_quant_mode"] == 1:
        # Undo C8 compensation using each result's own Q scale, checked above.
        # Apply the BF16 budget to the reconstructed projection in FP32.
        c_inv_scale = inputs["quant_scale_ckv"].cpu().float()
        query_rope *= outputs[2].cpu() / c_inv_scale
        query_rope_ref *= expected[2].cpu() / c_inv_scale
    _assert_mixed_accuracy(
        query_rope,
        query_rope_ref,
        dtype=torch.bfloat16,
        path="query_rope",
    )
    if inputs["query_norm_flag"]:
        norm, norm_ref = outputs[3].cpu().float(), expected[3].cpu().float()
        if inputs["weight_quant_mode"] == 3:
            scale = outputs[4].cpu().view(torch.uint8)
            scale_ref = expected[4].cpu().view(torch.uint8)
            assert torch.all(scale != 255), "MXFP8 scales must be finite"
            assert torch.equal(scale, scale_ref), "query_norm_scale: E8M0 mismatch"
            _assert_mixed_accuracy(
                norm, norm_ref, dtype=torch.float8_e4m3fn, path="query_norm"
            )
        elif inputs["weight_quant_mode"] == 2:
            assert_accuracy(
                norm, norm_ref, _INT8_QUERY_NORM_STANDARD, path="query_norm"
            )
            assert_accuracy(outputs[4], expected[4], path="query_norm_scale")
        else:
            assert_accuracy(norm, norm_ref, _BF16_STANDARD, path="query_norm")
    untouched = torch.ones(3 * page_size, dtype=torch.bool, device="cpu")
    slots = inputs["cache_index"].cpu()
    untouched[slots[slots >= 0]] = False
    for name in ("kv_cache", "kr_cache"):
        rows = _cache_rows(inputs[name], inputs["cache_mode"])
        rows_ref = _cache_rows(reference[name], inputs["cache_mode"])
        old_rows = _cache_rows(before[name], inputs["cache_mode"])
        # Byte comparison also covers FP8 without requiring an FP8 compare kernel.
        assert torch.equal(
            rows.view(torch.uint8)[untouched], old_rows.view(torch.uint8)[untouched]
        ), f"{name}: an unwritten slot changed"
        value, value_ref = rows.float(), rows_ref.float()
        if name == "kv_cache" and inputs["kv_cache_quant_mode"] == 1:
            # Both caches share the input scale. Only written slots contribute
            # to the match ratio; untouched slots are checked byte-for-byte above.
            _assert_mixed_accuracy(
                value[~untouched],
                value_ref[~untouched],
                dtype=torch.float8_e4m3fn,
                path=name,
            )
        else:
            assert_accuracy(value, value_ref, _BF16_STANDARD, path=name)


@pytest.mark.usefixtures("enable_arch22_internal_format")
@pytest.mark.parametrize(
    "shape,tokens,cache_mode,padding,weight_quant_mode,c8,use_rope",
    list(accuracy_cases()),
)
def test_mlaprolog_v4(
    shape,
    tokens,
    cache_mode,
    padding,
    weight_quant_mode,
    c8,
    use_rope,
):
    from sgl_kernel_npu.operators.attention import mlaprolog_v4

    inputs = generate_data(
        shape,
        tokens,
        weight_quant_mode,
        c8,
        cache_mode,
        not c8 and tokens != 17,
        padding=padding,
        use_rope=use_rope,
    )
    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    if checked:
        require_distinct_impls(mlaprolog_v4, impl, impl_ref)
        before, reference = _independent_cache(inputs), _independent_cache(inputs)
        pointers = [inputs[name].data_ptr() for name in ("kv_cache", "kr_cache")]
        expected = mlaprolog_v4(**reference, impl=impl_ref)
        outputs = mlaprolog_v4(**inputs, impl=impl)
        _accuracy(outputs, expected, inputs, reference, before)
        assert pointers == [
            inputs[name].data_ptr() for name in ("kv_cache", "kr_cache")
        ]
    else:
        require_test_impl(mlaprolog_v4, impl)
    # A CPU reference's transfer tasks are not an NPU compute baseline.
    if impl == "torch_native":
        return
    executor = make_executor(mlaprolog_v4, kwargs=inputs, accuracy_checked=checked)
    check_performance(
        executor, enable_perf=(tokens, shape[2], cache_mode, use_rope) in perf_shape
    )


@pytest.mark.usefixtures("enable_arch22_internal_format")
@pytest.mark.parametrize("shape", [(7168, 1536, 6, 128, 512, 64)], ids=["kimi-k3-tp16"])
@pytest.mark.parametrize("weight_quant_mode,c8", cache_cases)
@pytest.mark.parametrize("cache_mode", ["PA_BSND", "PA_NZ"])
@pytest.mark.parametrize("use_rope", [False, True], ids=["no-rope", "rope"])
def test_mlaprolog_v4_direct_cann_parity(
    shape, weight_quant_mode, c8, cache_mode, use_rope
):
    if get_skip_acc_check() is True:
        pytest.skip("Direct-CANN parity is disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.attention import mlaprolog_v4

    require_test_impl(mlaprolog_v4, "cann")
    from cann_ops_transformer import mla_prolog as cann_mla_prolog

    inputs = generate_data(
        shape, 4, weight_quant_mode, c8, cache_mode, not c8, use_rope=use_rope
    )
    if not use_rope:
        # Accuracy cases use None; parity also covers empty no-RoPE tensors.
        inputs["rope_sin"] = inputs["rmsnorm_gamma_cq"].new_empty(0)
        inputs["rope_cos"] = inputs["rmsnorm_gamma_cq"].new_empty(0)
    direct = _independent_cache(inputs)
    outputs = mlaprolog_v4(**inputs, impl="cann")
    expected = cann_mla_prolog(**direct)
    _check_contract(outputs, inputs)
    for actual, reference in zip(
        (*outputs, inputs["kv_cache"], inputs["kr_cache"]),
        (*expected, direct["kv_cache"], direct["kr_cache"]),
    ):
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        assert torch.equal(
            actual.cpu().view(torch.uint8), reference.cpu().view(torch.uint8)
        )
