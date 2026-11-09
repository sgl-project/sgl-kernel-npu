"""Validate packed regions against the selected reference, including skipped storage."""

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

# DeepSeek-V4: D=448+64, MXFP8 group=64, layout=1; synthetic batches/slots.
perf_shape = [(4096, 320), (4096, 512)]

# n, slots, d, group, mode, round, layout, e4m3, int64 slots, drop, seed, pattern
KV_CASES = [
    pytest.param(
        16, 16, 192, 0, 1, False, 1, True, False, True, 0, "normal", id="fp8-single"
    ),
    pytest.param(
        16,
        16,
        320,
        0,
        1,
        False,
        1,
        True,
        False,
        True,
        1,
        "normal",
        id="fp8-multi-scale",
    ),
    pytest.param(
        16, 16, 192, 0, 1, False, 1, False, True, True, 2, "normal", id="fp8-e5-int64"
    ),
    pytest.param(
        4096,
        4096,
        320,
        0,
        1,
        False,
        1,
        True,
        False,
        False,
        3,
        "normal",
        id="fp8-multi-core",
    ),
    pytest.param(
        64,
        128,
        512,
        0,
        1,
        False,
        1,
        False,
        False,
        False,
        4,
        "shuffle",
        id="fp8-partial-shuffled",
    ),
    pytest.param(
        16, 16, 192, 0, 1, False, 1, True, False, False, 5, "zero", id="fp8-zero"
    ),
    pytest.param(
        16, 16, 192, 64, 2, True, 1, True, False, True, 4, "normal", id="mx-group64"
    ),
    pytest.param(
        16,
        16,
        192,
        128,
        2,
        False,
        1,
        True,
        False,
        True,
        5,
        "normal",
        id="mx-group128-no-round",
    ),
    pytest.param(
        16, 16, 192, 64, 2, True, 1, False, False, True, 6, "normal", id="mx-e5"
    ),
    pytest.param(
        4096,
        4096,
        512,
        64,
        2,
        True,
        1,
        True,
        False,
        True,
        7,
        "normal",
        id="mx-multi-core",
    ),
    pytest.param(
        16, 16, 192, 128, 2, True, 2, True, False, True, 7, "normal", id="mx-block"
    ),
    pytest.param(
        17,
        17,
        320,
        64,
        2,
        True,
        2,
        True,
        False,
        True,
        9,
        "normal",
        id="mx-partial-block",
    ),
    pytest.param(
        17, 17, 320, 64, 2, True, 2, False, True, True, 10, "4d", id="mx-4d-int64"
    ),
    pytest.param(
        16,
        32,
        192,
        0,
        1,
        False,
        1,
        True,
        False,
        True,
        11,
        "stride",
        id="fp8-row-stride",
    ),
    pytest.param(
        16,
        32,
        512,
        64,
        2,
        True,
        1,
        True,
        True,
        True,
        12,
        "3d",
        id="deepseek-v4-mxfp8-slot-major-d512",
    ),
]


def generate_data(
    n,
    slots,
    d,
    group,
    mode,
    rounding,
    layout,
    e4,
    int64,
    drop,
    seed,
    pattern,
    device="npu",
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((n, d), dtype=torch.bfloat16, device="cpu", generator=generator)
    if pattern == "zero":
        x.zero_()
    mapping = torch.arange(n, device="cpu") % slots
    if pattern == "shuffle":
        mapping = torch.randperm(n, device="cpu", generator=generator) * 2
    if drop:
        mapping[0] = -1
    mapping = mapping.to(torch.int64 if int64 else torch.int32)
    q = d - 64
    scales = (q + group - 1) // group if mode == 2 else (q + 127) // 128
    if layout == 2:
        block = 8
        width = q + 128 + (scales + 7) // 8 * 8
        shape = ((slots + block - 1) // block, block, width)
        if pattern == "4d":
            shape = (shape[0], block, 1, width)
        byte = torch.full(shape, 91, dtype=torch.uint8, device=device)
    else:
        width = (q + 128 + scales * (1 if mode == 2 else 4) + 127) // 128 * 128
        byte = torch.full(
            (slots, width + (128 if pattern == "stride" else 0)),
            91,
            dtype=torch.uint8,
            device=device,
        )
    if pattern == "3d":
        byte = byte.unsqueeze(1)
    cache = byte.view(torch.float8_e4m3fn if e4 else torch.float8_e5m2)
    return cache, x.to(device), mapping.to(device), group, mode, rounding, layout


def _layout2_scale_columns(ids, block, value_width, scale_count):
    scale_width = (scale_count + 7) // 8 * 8
    start = block * value_width + (ids % block) * scale_width
    return start[:, None] + torch.arange(scale_width, device="cpu")


def _compare_cache(actual, reference, x, slots, group, mode, layout):
    actual = actual.squeeze(2) if actual.ndim == 4 else actual
    reference = reference.squeeze(2) if reference.ndim == 4 else reference
    a, r = actual.view(torch.uint8).cpu(), reference.view(torch.uint8).cpu()
    keep = torch.ones(a.shape, dtype=torch.bool, device="cpu")
    q = x.shape[1] - 64
    scale_count = (q + group - 1) // group if mode == 2 else (q + 127) // 128
    ids = slots.cpu().long()
    valid = ids != -1
    ids = ids[valid]
    if layout == 2:
        width = q + 128
        block = actual.shape[1]
        a, r, keep = [v.reshape(v.shape[0], -1) for v in (a, r, keep)]
        rows = ids // block
        cols = (ids % block)[:, None] * width + torch.arange(q, device="cpu")
        aq, rq = a[rows[:, None], cols], r[rows[:, None], cols]
        keep[rows[:, None], cols] = False
        scale_cols = _layout2_scale_columns(ids, block, width, scale_count)
        # Decode with reference exponents; keep checks the stored exponent bytes separately.
        scales = r[rows[:, None], scale_cols[:, :scale_count]]
        # Only scale_count bytes in a written slot's aligned scale row are defined.
        keep[rows[:, None], scale_cols[:, scale_count:]] = False
    else:
        a, r, keep = [v.reshape(v.shape[0], -1) for v in (a, r, keep)]
        off = 128 if mode == 2 else 0
        aq, rq = a[ids, off : off + q], r[ids, off : off + q]
        keep[ids, off : off + q] = False
        off = q + 128
        if mode == 2:
            scales = a[ids, off : off + scale_count]
        else:
            ac = a[ids, off : off + scale_count * 4].contiguous().view(torch.float32)
            scales = ac
            rc = r[ids, off : off + scale_count * 4].contiguous().view(torch.float32)
            assert_accuracy(
                ac, rc, {"torch.float32": AccuracyStandard(atol=1e-7, rtol=1e-5)}
            )
            keep[ids, off : off + scale_count * 4] = False
    quant = aq.contiguous().view(actual.dtype).float()
    assert_accuracy(
        quant,
        rq.contiguous().view(reference.dtype).float(),
        {
            "torch.float32": AccuracyStandard(
                atol=0.02, rtol=0.15 if mode == 1 else 0.25
            )
        },
    )
    assert_accuracy(a[keep], r[keep])

    if mode == 2:
        scales = torch.exp2(scales.float() - 127)
    expanded = scales.repeat_interleave(group if mode == 2 else 128, dim=1)[:, :q]
    restored = quant * expanded
    expected = x.cpu()[valid, :q].float()
    if mode == 2 and layout == 1:
        unsaturated = expected.abs() / expanded <= torch.finfo(actual.dtype).max
        assert unsaturated.any(), "test input must contain unsaturated MXFP8 values"
        restored, expected = restored[unsaturated], expected[unsaturated]
    torch.testing.assert_close(
        restored, expected, rtol=0.15 if mode == 1 else 0.25, atol=0.02
    )


@pytest.mark.parametrize(
    "n,slots,d,group,mode,rounding,layout,e4,int64,drop,seed,pattern", KV_CASES
)
def test_kv_compress_epilog(
    n, slots, d, group, mode, rounding, layout, e4, int64, drop, seed, pattern
):
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

    impl, impl_ref = get_test_impls()
    skip = bool(get_skip_acc_check())
    if skip:
        require_test_impl(kv_compress_epilog, impl)
    else:
        require_distinct_impls(kv_compress_epilog, impl, impl_ref)
    args = generate_data(
        n, slots, d, group, mode, rounding, layout, e4, int64, drop, seed, pattern
    )
    cache, x, mapping = args[:3]
    saved_x, saved_mapping = x.clone(), mapping.clone()
    if not skip:
        reference = cache.clone()
        assert kv_compress_epilog(reference, *args[1:], impl=impl_ref) is None
    pointer = cache.data_ptr()
    assert kv_compress_epilog(*args, impl=impl) is None
    assert cache.data_ptr() == pointer
    if not skip:
        _compare_cache(cache, reference, x, mapping, group, mode, layout)
    assert_accuracy(
        (x.view(torch.uint8), mapping), (saved_x.view(torch.uint8), saved_mapping)
    )
    executor = make_executor(
        kv_compress_epilog,
        args=args,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip,
    )
    check_performance(executor, enable_perf=(n, d) in perf_shape)


@pytest.mark.extra_checks
def test_fp8_layout2_rejected():
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

    require_test_impl(kv_compress_epilog, get_test_impls()[0])
    args = generate_data(1, 1, 192, 0, 1, False, 2, True, False, False, 0, "normal")
    with pytest.raises(RuntimeError, match="layout=2 only supports MXFP8"):
        kv_compress_epilog(*args, impl=get_test_impls()[0])
    if get_test_impls()[0] in ("ascendc", "auto"):
        with pytest.raises(RuntimeError, match="layout=2 only supports MXFP8"):
            torch.ops.npu.kv_compress_epilog(*args)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "name,index,value",
    [
        pytest.param("quant_mode", 4, -1, id="negative-mode"),
        pytest.param("quant_mode", 4, 0, id="zero-mode"),
        pytest.param("quant_mode", 4, 3, id="unknown-mode"),
        pytest.param("layout", 6, -1, id="negative-layout"),
        pytest.param("layout", 6, 0, id="zero-layout"),
        pytest.param("layout", 6, 3, id="unknown-layout"),
    ],
)
def test_kv_enum_inputs_rejected(name, index, value):
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

    args = list(
        generate_data(1, 1, 192, 128, 1, False, 1, True, False, False, 0, "normal")
    )
    args[index] = value
    saved = args[0].clone()
    impl, impl_ref = get_test_impls()
    for backend in dict.fromkeys((impl, impl_ref)):
        require_test_impl(kv_compress_epilog, backend)
        with pytest.raises(RuntimeError, match=f"{name} must be"):
            kv_compress_epilog(*args, impl=backend)
    if impl in ("ascendc", "auto"):
        with pytest.raises(RuntimeError, match=f"{name} must be"):
            torch.ops.npu.kv_compress_epilog(*args)
    assert_accuracy(args[0].view(torch.uint8), saved.view(torch.uint8))


@pytest.mark.extra_checks
def test_ascendc_graph_replay_with_distinct_tilings():
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

    require_test_impl(kv_compress_epilog, "ascendc")
    stream = torch.npu.Stream()
    captures = []
    # Keep both graphs live: the second capture must preserve the first graph's tiling.
    for layout, d, group, mode, pattern in [
        (1, 192, 0, 1, "stride"),
        (2, 320, 64, 2, "4d"),
    ]:
        args = generate_data(
            17, 32, d, group, mode, True, layout, True, True, True, 3, pattern
        )
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            kv_compress_epilog(*args, impl="ascendc")
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            kv_compress_epilog(*args, impl="ascendc")
        captures.append((graph, args))

    for shift in (1, 7):
        for graph, (cache, x, mapping, *attrs) in captures:
            with torch.npu.stream(stream):
                x.neg_()
                mapping.copy_(mapping.roll(shift))
                cache.view(torch.uint8).fill_(91)
                expected = cache.clone()
                kv_compress_epilog(expected, x, mapping, *attrs, impl="ascendc")
                graph.replay()
            stream.synchronize()
            actual_bytes = cache.view(torch.uint8).cpu().reshape(cache.shape[0], -1)
            expected_bytes = expected.view(torch.uint8).cpu().reshape_as(actual_bytes)
            if attrs[-1] == 2:
                block = cache.shape[1]
                q = x.shape[1] - 64
                group = attrs[0]
                count = (q + group - 1) // group
                ids = mapping.cpu().long()
                ids = ids[ids != -1]
                padding = _layout2_scale_columns(ids, block, q + 128, count)[:, count:]
                keep = torch.ones_like(actual_bytes, dtype=torch.bool)
                keep[(ids // block)[:, None], padding] = False
                assert_accuracy(actual_bytes[keep], expected_bytes[keep])
            else:
                assert_accuracy(actual_bytes, expected_bytes)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        ("x-dtype", "x must be BF16"),
        ("slot-dtype", "slot_mapping must be INT32 or INT64"),
        ("cache-dtype", "kv_compress_cache must be FP8"),
        ("headnum", "headnum"),
        ("stride", "block_stride"),
    ],
)
def test_kv_inputs_rejected(case, message):
    from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

    require_test_impl(kv_compress_epilog, get_test_impls()[0])
    args = list(
        generate_data(1, 1, 192, 128, 1, False, 1, True, False, False, 0, "normal")
    )
    if case == "x-dtype":
        args[1] = args[1].to(torch.float32)
    elif case == "slot-dtype":
        args[2] = args[2].to(torch.float32)
    elif case == "cache-dtype":
        args[0] = args[0].to(torch.float32)
    elif case == "headnum":
        args[0] = args[0].unsqueeze(1).unsqueeze(1).expand(-1, 1, 2, -1)
    else:
        args[0] = torch.empty((1, 128), dtype=args[0].dtype, device=args[0].device)
    impl = get_test_impls()[0]
    with pytest.raises(RuntimeError, match=message):
        kv_compress_epilog(*args, impl=impl)
    if impl in ("ascendc", "auto"):
        with pytest.raises(RuntimeError, match=message):
            torch.ops.npu.kv_compress_epilog(*args)
