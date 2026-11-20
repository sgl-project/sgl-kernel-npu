"""GLM DSA and DeepSeek MLA Prolog accuracy cases."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")

_PAGE_SIZE = 128
_LATENT = 512
_ROPE = 64
_TILE = 128


def generate_sparse_data(encoding, heads, context_length, topk):
    generator = torch.Generator(device="cpu").manual_seed(929)
    dtype = torch.bfloat16
    query = (
        torch.randn((1, heads, _LATENT + _ROPE), generator=generator, device="cpu")
        * 0.5
    ).to(dtype)
    # Keep the rotary contribution large enough to detect a dropped RoPE slice.
    query[..., _LATENT:] *= 4
    pages = context_length // _PAGE_SIZE
    page_shape = (pages, _PAGE_SIZE, 1)
    latent_shape = (*page_shape, _LATENT)
    if encoding == "int8":
        latent = torch.randint(
            -8, 9, latent_shape, generator=generator, device="cpu", dtype=torch.int8
        )
        scale_factor = 0.125
    else:
        latent = torch.randn(latent_shape, generator=generator, device="cpu").to(
            torch.float8_e4m3fn
        )
        scale_factor = 0.5
    rope = (
        torch.randn((*page_shape, _ROPE), generator=generator, device="cpu") * 2
    ).to(dtype)
    scales = (
        torch.rand((*page_shape, _LATENT // _TILE), generator=generator, device="cpu")
        * 0.75
        + 0.25
    ) * scale_factor
    # Match the SGLang DSA packed cache: 512 quantized bytes, 64 BF16 RoPE
    # values, then four FP32 scales per cached token.
    packed_bytes = torch.cat(
        (latent.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)),
        dim=-1,
    )
    packed = packed_bytes.npu().view(latent.dtype)
    page_table = torch.randperm(pages, generator=generator, device="cpu")
    selected = (
        torch.randperm(context_length, generator=generator, device="cpu")[:topk]
        .sort()
        .values
    )
    args = (
        query.npu(),
        packed,
        packed,
        selected.to(torch.int32).view(1, 1, topk).npu(),
        1.0 / 24,
        2,
        2,
    )
    kwargs = dict(
        block_table=page_table.to(torch.int32).unsqueeze(0).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([context_length], dtype=torch.int32),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=_TILE,
        rope_head_dim=_ROPE,
    )
    return args, kwargs


@pytest.mark.parametrize(
    "heads, context_length, topk",
    [pytest.param(32, 4096, 2048, id="glm52-dsa-paged-decode")],
)
def test_kv_quant_sparse_flash_attention(heads, context_length, topk):
    import torch_npu
    from sgl_kernel_npu.operators.attention import (
        kv_quant_sparse_flash_attention_torch_npu,
    )
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if not callable(getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)):
        pytest.skip("Installed TorchNPU lacks npu_kv_quant_sparse_flash_attention")
    # The GLM workload uses different cache encodings on the two generations.
    encoding = "int8" if get_hardware().generation is Generation.ARCH22 else "fp8"
    args, kwargs = generate_sparse_data(encoding, heads, context_length, topk)
    executor = check_accuracy(
        kv_quant_sparse_flash_attention_torch_npu,
        args,
        kwargs=kwargs,
        standard={"torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=3e-2)},
    )
    check_performance(executor)


@pytest.fixture
def allow_internal_weight_format():
    import torch_npu
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH22:
        yield
        return

    # An unset option uses the Atlas A2/A3 default of disabled.
    was_enabled = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    try:
        torch_npu.npu.config.allow_internal_format = True
        yield
    finally:
        torch_npu.npu.config.allow_internal_format = was_enabled


def generate_prolog_data(hidden, cq_rank, heads, head_dim):
    import torch_npu

    # BF16 NZ weights and the DeepSeek projection/absorption dimensions.
    dtype = torch.bfloat16
    token_x = torch.zeros((2, hidden), dtype=dtype)
    token_x[0, 0] = token_x[1, 1] = 1
    weight_dq = torch.zeros((hidden, cq_rank), dtype=dtype)
    weight_dkv_kr = torch.zeros((hidden, _LATENT + _ROPE), dtype=dtype)
    for weight in (weight_dq, weight_dkv_kr):
        weight[0].fill_(2)
        weight[1].fill_(-0.5)
    weight_uq_qr = torch.zeros((cq_rank, heads * (head_dim + _ROPE)), dtype=dtype)
    weight_uq_qr[0].fill_(1)
    weight_uk = (
        torch.eye(head_dim, _LATENT, dtype=dtype, device="cpu")
        .expand(heads, -1, -1)
        .contiguous()
        .npu()
    )
    gamma_cq = torch.ones(cq_rank, dtype=dtype)
    gamma_ckv = torch.ones(_LATENT, dtype=dtype)
    sin = torch.zeros((2, _ROPE), dtype=dtype)
    cos = torch.zeros((2, _ROPE), dtype=dtype)
    cos[0].fill_(1)
    sin[1].fill_(1)
    slots = [_PAGE_SIZE, _PAGE_SIZE - 1]
    kv_cache = torch.zeros((2, _PAGE_SIZE, 1, _LATENT), dtype=dtype)
    kr_cache = torch.zeros((2, _PAGE_SIZE, 1, _ROPE), dtype=dtype)
    args = (
        token_x,
        torch_npu.npu_format_cast(weight_dq, 29),
        torch_npu.npu_format_cast(weight_uq_qr, 29),
        weight_uk,
        torch_npu.npu_format_cast(weight_dkv_kr, 29),
        gamma_cq,
        gamma_ckv,
        sin,
        cos,
    )
    return args, dict(
        kv_cache=kv_cache,
        kr_cache=kr_cache,
        cache_index=torch.tensor(slots, dtype=torch.int64),
        cache_mode="PA_BSND",
        weight_quant_mode=0,
    )


@pytest.mark.parametrize(
    "hidden, cq_rank, heads, head_dim",
    [pytest.param(7168, 1536, 16, 128, id="deepseek-v3-tp8-bf16-paged-prolog")],
)
def test_mla_prolog_v3_paged_writes(
    allow_internal_weight_format, hidden, cq_rank, heads, head_dim
):
    import torch_npu
    from sgl_kernel_npu.operators.attention import mla_prolog_v3

    if not callable(getattr(torch_npu, "npu_mla_prolog_v3", None)):
        pytest.skip("Installed TorchNPU lacks npu_mla_prolog_v3")
    args, kwargs = generate_prolog_data(hidden, cq_rank, heads, head_dim)
    executor = check_accuracy(
        mla_prolog_v3,
        args,
        kwargs=kwargs,
        mutable_kwargs=("kv_cache", "kr_cache"),
        result_transform=lambda result: result[:2],
        standard={"torch.bfloat16": AccuracyStandard(atol=1e-5, rtol=1.6e-2)},
    )
    check_performance(executor)
