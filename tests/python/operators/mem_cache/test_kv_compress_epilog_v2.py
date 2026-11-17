"""Validate written rows and pages of the MXFP8/MXFP4 KV writer."""

import pytest

torch = pytest.importorskip("torch")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.operators.utils.cann import detect_cann_version

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


def _unavailable():
    """Skip outside the released CANN 9.2 channel instead of failing there."""
    try:
        return not detect_cann_version().startswith("9.2.")
    except RuntimeError:
        return True


pytestmark = pytest.mark.skipif(_unavailable(), reason="requires CANN 9.2.x")

# DeepSeek-V4.1 full-width KV rows: D=512, MXFP8-g32 or MXFP4-g16; synthetic batches.
perf_shape = [(4096, 512)]

_MXFP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# n, slots, d, mode, group, cache, round, layout, block, slot_dtype, seed, pattern
KV_CASES = [
    pytest.param(
        4096,
        4096,
        512,
        "mxfp8_bf16",
        32,
        "e4m3",
        True,
        2,
        0,
        "int32",
        0,
        "normal",
        id="deepseek-v4-1-mxfp8-g32-multicore",
    ),
    pytest.param(
        4096,
        4096,
        512,
        "mxfp4_bf16",
        16,
        "uint8",
        True,
        2,
        0,
        "int32",
        1,
        "normal",
        id="deepseek-v4-1-mxfp4-g16-multicore",
    ),
    pytest.param(
        128,
        256,
        512,
        "mxfp8_bf16",
        32,
        "e4m3",
        True,
        4,
        128,
        "int32",
        2,
        "normal",
        id="deepseek-v4-1-mxfp8-paged",
    ),
    pytest.param(
        128,
        256,
        512,
        "mxfp4_bf16",
        16,
        "uint8",
        True,
        4,
        128,
        "int64",
        3,
        "normal",
        id="deepseek-v4-1-mxfp4-paged-int64",
    ),
    pytest.param(
        16,
        16,
        512,
        "mxfp8_bf16",
        32,
        "e5m2",
        False,
        2,
        0,
        "int32",
        4,
        "normal",
        id="edge-mxfp8-e5m2-no-round",
    ),
    pytest.param(
        16,
        16,
        32,
        "mxfp8_bf16",
        32,
        "e4m3",
        True,
        2,
        0,
        "int32",
        5,
        "normal",
        id="edge-minimum-width",
    ),
    pytest.param(
        16,
        16,
        48,
        "mxfp4_bf16",
        16,
        "uint8",
        True,
        2,
        0,
        "int32",
        6,
        "normal",
        id="edge-width-48-g16",
    ),
    pytest.param(
        16,
        16,
        48,
        "mxfp4_bf16",
        16,
        "uint8",
        True,
        4,
        32,
        "int32",
        7,
        "normal",
        id="edge-paged-unaligned-data",
    ),
    pytest.param(
        16,
        16,
        512,
        "mxfp8_bf16",
        32,
        "e4m3",
        True,
        2,
        0,
        "int32",
        8,
        "zero",
        id="edge-zero-input",
    ),
    pytest.param(
        64,
        128,
        512,
        "mxfp4_bf16",
        32,
        "uint8",
        True,
        2,
        0,
        "int64",
        9,
        "shuffle",
        id="edge-mxfp4-g32-shuffled",
    ),
]

_CACHE_DTYPES = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _widths(d, mode, group, layout):
    """Return (data_width, token_width, cache_col) for the row layout."""
    data_width = d if mode == "mxfp8_bf16" else d // 2
    token_width = data_width + (d // group) * 2
    cache_col = token_width if layout == 4 else (token_width + 31) // 32 * 32
    return data_width, token_width, cache_col


def generate_data(
    n,
    slots,
    d,
    mode,
    group,
    cache,
    round_scale,
    layout,
    block,
    slot_dtype,
    seed,
    pattern,
):
    # Host-side construction keeps the seeded generator on CPU (see
    # test_quant_flash_attn.py); the operator inputs then move to the NPU.
    device = torch.get_default_device()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((n, d), dtype=torch.bfloat16, device="cpu", generator=generator)
    if pattern == "zero":
        x.zero_()
    mapping = torch.arange(n, device="cpu") % slots
    if pattern == "shuffle":
        mapping = torch.randperm(n, device="cpu", generator=generator) * 2 % slots
    mapping = mapping.to(torch.int64 if slot_dtype == "int64" else torch.int32)
    _, _, cache_col = _widths(d, mode, group, layout)
    if layout == 4:
        pages = (slots + block - 1) // block
        byte = torch.full(
            (pages, block, 1, cache_col), 91, dtype=torch.uint8, device=device
        )
    else:
        byte = torch.full((slots, cache_col + 32), 91, dtype=torch.uint8, device=device)
    if cache != "uint8":
        byte = byte.view(_CACHE_DTYPES[cache])
    kwargs = dict(
        quant_group_size=group,
        quant_mode=mode,
        round_scale=round_scale,
        x_scale=1.0,
    )
    return byte, x.to(device), mapping.to(device), kwargs


def _decode_mxfp4(packed, rows, d):
    low = packed & 0xF
    high = packed >> 4
    levels = _MXFP4_LEVELS.to(packed.device)

    def value(nibble):
        magnitude = levels[(nibble & 7).long()]
        return torch.where((nibble & 8) > 0, -magnitude, magnitude)

    return torch.stack((value(low), value(high)), dim=-1).reshape(rows, d)


def _scale_words(words):
    """Little-endian scale bytes -> float32 exponents (2^(field-127))."""
    value = words[..., 0].to(torch.int64) | (words[..., 1].to(torch.int64) << 8)
    return torch.exp2((value >> 7).float() - 127)


def _compare_cache(actual, reference, x, slots, mode, group):
    a = actual.view(torch.uint8).cpu()
    r = reference.view(torch.uint8).cpu()
    keep = torch.ones(a.shape, dtype=torch.bool)
    a, r, keep = [v.reshape(v.shape[0], -1) for v in (a, r, keep)]
    data_width, _, _ = _widths(x.shape[1], mode, group, 4 if actual.ndim == 4 else 2)
    scale_count = x.shape[1] // group

    ids = slots.cpu().long()
    selected = ids[ids >= 0]
    if actual.ndim == 4:
        row_index = (selected // actual.shape[1])[:, None]
        base = ((selected % actual.shape[1]) * (data_width + scale_count * 2))[:, None]
    else:
        row_index = selected[:, None]
        base = torch.zeros_like(selected)[:, None]
    data_cols = base + torch.arange(data_width)
    scale_cols = base + data_width + torch.arange(scale_count * 2)
    keep[row_index, data_cols] = False
    keep[row_index, scale_cols] = False

    if mode == "mxfp8_bf16":
        actual_values = a[row_index, data_cols].contiguous().view(actual.dtype).float()
        reference_values = (
            r[row_index, data_cols].contiguous().view(reference.dtype).float()
        )
        actual_scales = (
            a[row_index, scale_cols].contiguous().view(torch.bfloat16).float()
        )
        reference_scales = (
            r[row_index, scale_cols].contiguous().view(torch.bfloat16).float()
        )
        standard = AccuracyStandard(atol=0.02, rtol=0.15)
    else:
        actual_values = _decode_mxfp4(
            a[row_index, data_cols].contiguous(), len(selected), x.shape[1]
        )
        reference_values = _decode_mxfp4(
            r[row_index, data_cols].contiguous(), len(selected), x.shape[1]
        )
        actual_scales = _scale_words(
            a[row_index, scale_cols].contiguous().view(len(selected), scale_count, 2)
        )
        reference_scales = _scale_words(
            r[row_index, scale_cols].contiguous().view(len(selected), scale_count, 2)
        )
        standard = AccuracyStandard(atol=0.02, rtol=0.3)

    expanded = actual_scales.repeat_interleave(group, dim=1)[:, : x.shape[1]]
    restored = actual_values * expanded
    assert_accuracy(restored, x.cpu()[ids >= 0].float(), {"torch.float32": standard})
    assert_accuracy(reference_values, actual_values, {"torch.float32": standard})
    assert_accuracy(
        actual_scales,
        reference_scales,
        {"torch.float32": AccuracyStandard(atol=1e-7, rtol=1e-5)},
    )
    # Unwritten bytes and zero padding must stay byte-identical.
    assert_accuracy(a[keep], r[keep])


@pytest.mark.parametrize(
    "n,slots,d,mode,group,cache,round_scale,layout,block,slot_dtype,seed,pattern",
    [param.values for param in KV_CASES],
)
def test_kv_compress_epilog_v2(
    n,
    slots,
    d,
    mode,
    group,
    cache,
    round_scale,
    layout,
    block,
    slot_dtype,
    seed,
    pattern,
):
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog_v2

    impl, impl_ref = get_test_impls()
    skip = bool(get_skip_acc_check())
    if skip:
        require_test_impl(kv_compress_epilog_v2, impl)
    else:
        require_distinct_impls(kv_compress_epilog_v2, impl, impl_ref)
    cache, x, mapping, kwargs = generate_data(
        n,
        slots,
        d,
        mode,
        group,
        cache,
        round_scale,
        layout,
        block,
        slot_dtype,
        seed,
        pattern,
    )
    saved_x, saved_mapping = x.clone(), mapping.clone()
    reference = None
    if not skip:
        reference = cache.clone()
        kv_compress_epilog_v2(reference, x, mapping, impl=impl_ref, **kwargs)
    pointer = cache.data_ptr()
    assert kv_compress_epilog_v2(cache, x, mapping, impl=impl, **kwargs) is None
    assert cache.data_ptr() == pointer
    if not skip:
        _compare_cache(cache, reference, x, mapping, mode, group)
    assert_accuracy(
        (x.view(torch.uint8), mapping),
        (saved_x.view(torch.uint8), saved_mapping),
    )
    executor = make_executor(
        kv_compress_epilog_v2,
        args=(cache, x, mapping),
        kwargs=kwargs,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip,
    )
    check_performance(executor, enable_perf=(n, d) in perf_shape)


@pytest.mark.extra_checks
def test_invalid_mode_rejected():
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog_v2

    require_test_impl(kv_compress_epilog_v2, get_test_impls()[0])
    cache, x, mapping, _ = generate_data(
        1, 1, 512, "mxfp8_bf16", 32, "e4m3", True, 2, 0, "int32", 0, "normal"
    )
    # The binding raises RuntimeError; the eager reference raises ValueError.
    with pytest.raises((ValueError, RuntimeError), match="quant_mode should be"):
        kv_compress_epilog_v2(cache, x, mapping, quant_mode="fp8_bf16")
