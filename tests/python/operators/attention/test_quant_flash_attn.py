"""A5 MXFP8 QFA numerical checks and public CANN interface parity.

Qwen3.6-27B full attention has 24 Q heads, 4 KV heads and head_dim=256.
TP1/TP2 cases use the corresponding per-rank heads and SGLang page_size=512.
Source: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json
Lengths and batch sizes are benchmark workloads; tensor values are synthetic,
not captured model activations. TP2 is a per-rank operator test, not a TP launch.
"""

import math

import pytest
import torch

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Load the kernel package before configure_npu initializes the device, so its
# bundled CANN vendor path is registered before the runtime reads it.
pytest.importorskip("torch_npu")
import sgl_kernel_npu  # noqa: F401

# ((local Q heads, local KV heads, head dimension, page size), Q lengths, KV lengths).
# Prefill: BS1, 8K/16K. Decode: BS1..64 with 8K/16K cached tokens per request.
perf_shape = [
    *[
        ((24 // tp, 4 // tp, 256, 512), (seq_len,), (seq_len,))
        for tp in (1, 2)
        for seq_len in (8192, 16384)
    ],
    *[
        ((24 // tp, 4 // tp, 256, 512), (1,) * batch_size, (kv_len,) * batch_size)
        for tp in (1, 2)
        for kv_len in (8192, 16384)
        for batch_size in (1, 2, 4, 8, 16, 32, 64)
    ],
]
qwen_cases = [
    pytest.param(
        shape,
        q_lens,
        kv_lens,
        "N2TGD" if q_lens[0] == 1 else "TND",
        0 if q_lens[0] == 1 else 3,
        1.0,
        False,
        "static",
        id=(
            f"qwen3.6-27b-tp{24 // shape[0]}-"
            f"{'decode' if q_lens[0] == 1 else 'prefill'}-"
            f"bs{len(q_lens)}-q{q_lens[0]}-kv{kv_lens[0]}"
        ),
    )
    for shape, q_lens, kv_lens in perf_shape
]


def generate_data(
    shape, q_lens, kv_lens, scale_layout, mask_mode, p_scale, lse, v_mode
):
    """Build independent FP8 data with varying power-of-two descales.

    CPU construction avoids relying on the quantization kernels under test.
    The test's default NPU device is supplied by conftest. Physical pages are
    permuted, and every tensor passed to QFA has nonzero strides.
    """
    device = torch.get_default_device()
    generator = torch.Generator(device="cpu").manual_seed(42)
    nq, nkv, dim, page_size = shape
    total_q = sum(q_lens)
    pages_per_request = [math.ceil(length / page_size) for length in kv_lens]
    total_pages = max(1, sum(pages_per_request))
    max_pages = max(1, max(pages_per_request))
    table = torch.zeros((len(q_lens), max_pages), dtype=torch.int32, device="cpu")
    permutation = torch.randperm(total_pages, generator=generator, device="cpu")
    offset = 0
    for request, pages in enumerate(pages_per_request):
        table[request, :pages] = permutation[offset : offset + pages]
        offset += pages

    def scales(scale_shape):
        # Exponents 122..125 represent 1/32..1/4: all channels remain finite,
        # and omitting descales creates a substantial numerical error.
        return torch.randint(
            122, 126, scale_shape, generator=generator, device="cpu", dtype=torch.uint8
        )

    def quantized(data_shape, expanded_scale):
        values = torch.randn(data_shape, generator=generator, device="cpu")
        return (values / torch.exp2(expanded_scale.float() - 127)).to(
            torch.float8_e4m3fn
        )

    q_scale = scales((total_q, nq, dim // 32))
    k_scale = scales((total_pages, nkv, page_size, dim // 32))
    v_shape = (total_pages, nkv, page_size // 64, dim, 2)
    if v_mode == "static":
        # ModelSlim repeats one per-head/channel scale across all token groups.
        v_scale = scales((1, nkv, 1, dim, 1)).expand(v_shape).clone()
    elif v_mode == "grouped":
        # Valid QFA input with distinct even/odd sequence-group scales.
        v_scale = scales(v_shape)
    else:
        raise ValueError(v_mode)
    q = quantized((total_q, nq, dim), q_scale.repeat_interleave(32, dim=-1))
    k = quantized(
        (total_pages, nkv, page_size, dim), k_scale.repeat_interleave(32, dim=-1)
    )
    v_expanded = (
        v_scale.permute(0, 1, 2, 4, 3)
        .reshape(total_pages, nkv, page_size // 32, dim)
        .repeat_interleave(32, dim=2)
    )
    v = quantized((total_pages, nkv, page_size, dim), v_expanded)
    q_scale = q_scale.reshape(total_q, nq, dim // 64, 2)
    if scale_layout == "N2TGD":
        q_scale = (
            q_scale.reshape(total_q, nkv, nq // nkv, dim // 64, 2)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
    k_scale = k_scale.reshape(total_pages, nkv, page_size, dim // 64, 2)
    args = (
        q.to(device),
        k.to(device),
        v.to(device),
        q_scale.to(device).view(torch.float8_e8m0fnu),
        k_scale.to(device).view(torch.float8_e8m0fnu),
        v_scale.to(device).view(torch.float8_e8m0fnu),
        1,
    )
    cu_q = torch.tensor([0, *q_lens], dtype=torch.int32, device="cpu").cumsum(
        0, dtype=torch.int32
    )
    shared_kwargs = dict(
        cu_seqlens_q=cu_q.to(device),
        seqused_kv=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        max_seqlen_q=q_lens[0] if len(set(q_lens)) == 1 else -1,
        mask_mode=mask_mode,
        layout_q="TND",
        layout_q_descale=scale_layout,
        layout_kv="PA_BNBD",
        layout_out="TND",
    )
    kwargs = dict(
        **shared_kwargs,
        block_table=table.to(device),
        p_scale=torch.tensor([p_scale], dtype=torch.float32, device=device),
        attn_mask=(
            torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=device), 1)
            if mask_mode == 3
            else None
        ),
        softmax_scale=dim**-0.5,
        return_softmax_lse=lse,
    )
    return args, kwargs, (nq, nkv, dim, 1), shared_kwargs


@pytest.mark.parametrize(
    "shape,q_lens,kv_lens,scale_layout,mask_mode,p_scale,lse,v_mode",
    qwen_cases
    + [
        pytest.param(
            (16, 4, 64, 128),
            (1, 1),
            (1, 129),
            "N2TGD",
            0,
            1.0,
            False,
            "static",
            id="edge-decode-d64-page-tail",
        ),
        pytest.param(
            (16, 2, 128, 512),
            (1, 1, 1, 1),
            (513, 1024, 0, 127),
            "N2TGD",
            0,
            1.0,
            True,
            "static",
            id="edge-decode-d128-zero-kv-padding",
        ),
        pytest.param(
            (12, 2, 256, 512),
            (1, 1, 1, 1),
            (513, 1024, 2048, 127),
            "N2TGD",
            0,
            1.0,
            False,
            "static",
            id="edge-decode-d256",
        ),
        pytest.param(
            (16, 2, 128, 512),
            (129, 17),
            (129, 17),
            "TND",
            3,
            1.0,
            True,
            "static",
            id="edge-ragged-prefill-d128",
        ),
        pytest.param(
            (12, 2, 256, 512),
            (33, 17),
            (1025, 529),
            "TND",
            3,
            16.0,
            True,
            "static",
            id="edge-prefix-chunk-d256-nonunit-p",
        ),
        pytest.param(
            (12, 2, 256, 512),
            (4, 4),
            (513, 1027),
            "N2TGD",
            3,
            1.0,
            False,
            "static",
            id="edge-verify-d256",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (1,),
            (1,),
            "TND",
            0,
            16.0,
            False,
            "static",
            id="edge-minimum-single-head",
        ),
        pytest.param(
            (4, 2, 64, 128),
            (3, 5),
            (65, 257),
            "TND",
            3,
            16.0,
            True,
            "grouped",
            id="edge-v-scale-even-odd-groups",
        ),
    ],
)
def test_quant_flash_attn(
    shape, q_lens, kv_lens, scale_layout, mask_mode, p_scale, lse, v_mode
):
    from sgl_kernel_npu.operators.attention import (
        quant_flash_attn,
        quant_flash_attn_metadata,
    )

    require_test_impl(quant_flash_attn_metadata, "cann_ops_transformer")
    args, kwargs, meta_args, meta_kwargs = generate_data(
        shape, q_lens, kv_lens, scale_layout, mask_mode, p_scale, lse, v_mode
    )
    metadata = quant_flash_attn_metadata(*meta_args, **meta_kwargs)
    assert metadata.dtype == torch.int32
    assert metadata.device == args[0].device
    assert metadata.ndim == 2 and metadata.shape[0] == 2
    kwargs["metadata"] = metadata
    inputs = [value for value in (*args, *kwargs.values()) if torch.is_tensor(value)]
    snapshots = (
        [value.view(torch.uint8).clone() for value in inputs]
        if get_skip_acc_check() is not True
        else []
    )
    executor = check_accuracy(
        quant_flash_attn,
        args=args,
        kwargs=kwargs,
        # FP32 LSE is reduced differently by CANN's parallel softmax. Keep
        # the repository BF16 standard; this initial LSE tolerance requires
        # confirmation on A5, just like the shared accuracy standards.
        standard={"torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3)},
    )
    for value, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(value.view(torch.uint8), snapshot, atol=0, rtol=0)
    check_performance(executor, enable_perf=(shape, q_lens, kv_lens) in perf_shape)


@pytest.mark.parametrize(
    "shape,q_lens,kv_lens,scale_layout,mask_mode,p_scale,lse,v_mode",
    qwen_cases,
)
def test_quant_flash_attn_direct_cann_parity(
    shape, q_lens, kv_lens, scale_layout, mask_mode, p_scale, lse, v_mode
):
    """Check both adapters against direct CANN using the plan's consumer.

    Metadata contains an opaque schedule with potentially unused storage;
    compare the attention results, not the unused bytes of two schedules.
    Independent numerical correctness is covered by test_quant_flash_attn.
    """
    from sgl_kernel_npu.operators.attention import (
        quant_flash_attn,
        quant_flash_attn_metadata,
    )

    require_test_impl(quant_flash_attn_metadata, "cann_ops_transformer")
    from cann_ops_transformer.ops import quant_flash_attn as direct_qfa
    from cann_ops_transformer.ops import quant_flash_attn_metadata as direct_metadata

    args, kwargs, meta_args, meta_kwargs = generate_data(
        shape, q_lens, kv_lens, scale_layout, mask_mode, p_scale, lse, v_mode
    )
    checked = get_skip_acc_check() is not True
    if checked:
        metadata = quant_flash_attn_metadata(
            *meta_args, **meta_kwargs, impl="cann_ops_transformer"
        )
        expected_metadata = direct_metadata(*meta_args, **meta_kwargs)
        assert metadata.shape == expected_metadata.shape
        actual = quant_flash_attn(
            *args, **kwargs, metadata=metadata, impl="cann_ops_transformer"
        )
        expected = direct_qfa(*args, **kwargs, metadata=expected_metadata)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
    executor = make_executor(
        quant_flash_attn_metadata,
        args=meta_args,
        kwargs=meta_kwargs,
        impl="cann_ops_transformer",
        impl_ref="direct_cann_consumer",
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=(shape, q_lens, kv_lens) in perf_shape)
