"""A5 KV writer byte accuracy for the selected 608B or 640B cache encoding."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

perf_shape = [128]


@pytest.mark.parametrize("tokens", [1, 128], ids=["synthetic-decode", "synthetic-page"])
def test_kv_compress_epilog(tokens):
    from sgl_kernel_npu.operators.kvcacheio import (
        get_npu_kv_quant_path,
        kv_compress_epilog,
    )
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("mixed-quant reader/writer requires A5")
    torch.manual_seed(17)
    path = get_npu_kv_quant_path()
    x = torch.randn(tokens, 512, dtype=torch.bfloat16)
    slots = torch.arange(tokens, dtype=torch.int32)
    if tokens > 1:
        slots[-1] = -1
    # CANN 9.1 cannot zero FP8 directly; view zeroed bytes as the packed cache.
    cache = torch.zeros(1, 128, 1, path.packed_dim(448, 64), dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    kwargs = dict(
        quant_group_size=64,
        quant_mode=path.writer_quant_mode,
        round_scale=True,
        x_scale=1.0,
    )
    executor = make_executor(kv_compress_epilog, args=(cache, x, slots), kwargs=kwargs)
    if get_skip_acc_check() is True:
        require_test_impl(kv_compress_epilog, executor.impl)
    else:
        require_distinct_impls(kv_compress_epilog, executor.impl, executor.impl_ref)
        expected = kv_compress_epilog(
            cache.view(torch.uint8).clone().view(torch.float8_e4m3fn),
            x,
            slots,
            impl=executor.impl_ref,
            **kwargs,
        )
        actual = executor()
        # RoPE BF16 + NoPE FP8 + seven scales; alignment padding is unspecified.
        semantic_bytes = 576 + 7 * path.scale_itemsize
        assert_accuracy(
            actual.view(torch.uint8)[..., :semantic_bytes],
            expected.view(torch.uint8)[..., :semantic_bytes],
        )
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=tokens in perf_shape)
