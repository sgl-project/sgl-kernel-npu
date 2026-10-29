"""Kimi K3 no-RoPE MLAProlog V4: C16 and ModelSlim static C8 cache contracts.

Dimensions follow moonshotai/Kimi-K3 config.json, with 96 Q heads sharded
over attention TP sizes 1/2/4/8/16/32. Tensor values are generated, not loaded
from a checkpoint. Weights and scales follow SGLang's MLAProlog input layouts.
Source: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json
The independent torch_native implementation is an accuracy reference on CPU;
its transfers must not be reported as an NPU performance baseline.
"""

from functools import partial

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
from tests.python.utils.performance import benchmark_npu, check_performance
from tests.python.utils.standards import AccuracyStandard

# (hidden, query_lora_rank, local_heads, query_head_dim, kv_lora_rank, r_dim).
shapes = [(7168, 1536, heads, 128, 512, 64) for heads in (96, 48, 24, 12, 6, 3)]
token_counts = [1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 31, 32, 40, 48, 56, 63, 64]
page_size = 128
# Representative attention TP4/TP16 shapes; other cases check accuracy only.
perf_shape = [
    (tokens, heads, "PA_BSND") for heads in (24, 6) for tokens in (1, 4, 16, 32, 64)
]
cache_cases = [(0, False), (3, False), (3, True)]
_BF16_STANDARD = {"torch.float32": AccuracyStandard(atol=5e-2, rtol=5e-2)}
_QUERY_SCALE_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-7, rtol=1e-2)}


def generate_data(
    shape, tokens, weight_quant_mode, c8, cache_mode, query_norm_flag, *, padding=False
):
    """Create real NZ weights and distinct, nonconsecutive paged cache slots."""
    import torch_npu

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
    if weight_quant_mode == 3:
        for name, scale_name in (
            ("token_x", "dequant_scale_x"),
            ("weight_dq", "dequant_scale_w_dq"),
            ("weight_uq_qr", "dequant_scale_w_uq_qr"),
            ("weight_dkv_kr", "dequant_scale_w_dkv_kr"),
        ):
            is_weight = name != "token_x"
            value = inputs[name].T.contiguous() if is_weight else inputs[name]
            quantized, scales = torch_npu.npu_dynamic_mx_quant(
                value, axis=-1, dst_type=torch.float8_e4m3fn, block_size=32
            )
            inputs[name] = quantized.T.contiguous() if is_weight else quantized
            inputs[scale_name] = (
                scales.contiguous()
                .reshape(value.shape[0], -1)
                .view(torch.float8_e8m0fnu)
            )
    for name in ("weight_dq", "weight_uq_qr", "weight_dkv_kr"):
        inputs[name] = torch_npu.npu_format_cast(inputs[name], 29)
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


def _mx_scale(scales):
    exponent = scales.cpu().view(torch.uint8)
    assert torch.all(exponent != 255), "MXFP8 scales must be finite"
    return torch.pow(2.0, exponent.float() - 127)


def _check_contract(outputs, inputs):
    assert isinstance(outputs, tuple) and len(outputs) == 5
    tokens = inputs["token_x"].shape[0]
    rank = inputs["weight_dq"].shape[-1]
    heads, _, latent = inputs["weight_uk"].shape
    r_dim = inputs["kr_cache"].shape[-1]
    c8 = inputs["kv_cache_quant_mode"] == 1
    mode3 = inputs["weight_quant_mode"] == 3
    norm = inputs["query_norm_flag"]
    expected = (
        ((tokens, heads, latent), torch.float8_e4m3fn if c8 else torch.bfloat16),
        ((tokens, heads, r_dim), torch.bfloat16),
        ((tokens, heads, 1) if c8 else (0,), torch.float32),
        ((tokens, rank) if norm else (0,), inputs["weight_uq_qr"].dtype),
        (
            (tokens, rank // 32) if mode3 and norm else (0,),
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
        query *= outputs[2].cpu()
        query_ref *= expected[2].cpu()
    assert_accuracy(query, query_ref, _BF16_STANDARD, path="query")
    assert_accuracy(outputs[1], expected[1], path="query_rope")
    if inputs["query_norm_flag"]:
        norm, norm_ref = outputs[3].cpu().float(), expected[3].cpu().float()
        if inputs["weight_quant_mode"] == 3:
            norm = (
                norm.reshape(norm.shape[0], -1, 32) * _mx_scale(outputs[4])[..., None]
            )
            norm_ref = (
                norm_ref.reshape(norm_ref.shape[0], -1, 32)
                * _mx_scale(expected[4])[..., None]
            )
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
            multiplier = inputs["quant_scale_ckv"].cpu()
            value, value_ref = value / multiplier, value_ref / multiplier
        assert_accuracy(value, value_ref, _BF16_STANDARD, path=name)


@pytest.mark.parametrize(
    "shape", shapes, ids=[f"kimi-k3-tp{tp}" for tp in (1, 2, 4, 8, 16, 32)]
)
@pytest.mark.parametrize(
    "tokens,padding", [(tokens, False) for tokens in token_counts] + [(5, True)]
)
@pytest.mark.parametrize("cache_mode", ["PA_BSND", "PA_NZ"])
@pytest.mark.parametrize("weight_quant_mode,c8", cache_cases)
def test_mla_prolog_norope(
    shape, tokens, cache_mode, padding, weight_quant_mode, c8, tmp_path, record_property
):
    from sgl_kernel_npu.operators.attention import mla_prolog_norope

    inputs = generate_data(
        shape,
        tokens,
        weight_quant_mode,
        c8,
        cache_mode,
        not c8 and tokens != 17,
        padding=padding,
    )
    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    if checked:
        require_distinct_impls(mla_prolog_norope, impl, impl_ref)
        before, reference = _independent_cache(inputs), _independent_cache(inputs)
        pointers = [inputs[name].data_ptr() for name in ("kv_cache", "kr_cache")]
        expected = mla_prolog_norope(**reference, impl=impl_ref)
        outputs = mla_prolog_norope(**inputs, impl=impl)
        _accuracy(outputs, expected, inputs, reference, before)
        assert pointers == [
            inputs[name].data_ptr() for name in ("kv_cache", "kr_cache")
        ]
    else:
        require_test_impl(mla_prolog_norope, impl)
    # A CPU reference's transfer tasks are not an NPU compute baseline.
    if impl == "torch_native":
        return
    executor = make_executor(mla_prolog_norope, kwargs=inputs, accuracy_checked=checked)
    measured = check_performance(
        executor, enable_perf=(tokens, shape[2], cache_mode) in perf_shape
    )
    if measured is not None:
        from cann_ops_transformer import mla_prolog as cann_mla_prolog

        # Repeated writes use the same valid slots. This compares device task
        # durations, not Python dispatch overhead or end-to-end host latency.
        direct_us = benchmark_npu(
            partial(cann_mla_prolog, **inputs), tmp_path / "direct_cann"
        )
        record_property("direct_cann_average_device_task_us", direct_us)
        print(f"Direct CANN average device task duration: {direct_us:.3f} us")


@pytest.mark.parametrize("shape", [(7168, 1536, 6, 128, 512, 64)], ids=["kimi-k3-tp16"])
@pytest.mark.parametrize("weight_quant_mode,c8", cache_cases)
@pytest.mark.parametrize("cache_mode", ["PA_BSND", "PA_NZ"])
def test_mla_prolog_norope_direct_cann_parity(shape, weight_quant_mode, c8, cache_mode):
    if get_skip_acc_check() is True:
        pytest.skip("Direct-CANN parity is disabled by --skip-acc-check=true")
    from sgl_kernel_npu.operators.attention import mla_prolog_norope

    require_test_impl(mla_prolog_norope, "cann")
    from cann_ops_transformer import mla_prolog as cann_mla_prolog

    inputs = generate_data(shape, 4, weight_quant_mode, c8, cache_mode, not c8)
    # Cover empty no-RoPE arguments here; accuracy cases use None in both layouts.
    inputs["rope_sin"] = inputs["rmsnorm_gamma_cq"].new_empty(0)
    inputs["rope_cos"] = inputs["rmsnorm_gamma_cq"].new_empty(0)
    direct = _independent_cache(inputs)
    outputs = mla_prolog_norope(**inputs, impl="cann")
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


@pytest.mark.parametrize("impl", ["auto", "cann", "torch_native"])
@pytest.mark.parametrize("rope_arg", ["rope_sin", "rope_cos"])
def test_mla_prolog_norope_rejects_rope(impl, rope_arg):
    from sgl_kernel_npu.operators.attention import mla_prolog_norope

    # Rejection must precede backend dispatch and any cache access.
    unused_inputs = [torch.empty(0)] * 9
    with pytest.raises(
        ValueError, match="requires rope_sin and rope_cos to be None or empty"
    ):
        mla_prolog_norope(*unused_inputs, **{rope_arg: torch.ones(1)}, impl=impl)
