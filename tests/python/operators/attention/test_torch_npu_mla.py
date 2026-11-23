"""DeepSeek-V3.2 packed DSA and DeepSeek-V2 MLA Prolog accuracy cases."""

import math

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")

_PAGE_SIZE = 128
_LATENT = 512
_ROPE = 64
_TILE = 128


def generate_sparse_data(heads, context_length, topk):
    generator = torch.Generator(device="cpu").manual_seed(929)
    dtype = torch.bfloat16
    query = (
        torch.randn((1, heads, _LATENT + _ROPE), generator=generator, device="cpu")
        * 0.5
    ).to(dtype)
    # Keep the rotary contribution large enough to detect a dropped RoPE slice.
    query[..., _LATENT:] *= 4
    pages = (context_length + _PAGE_SIZE - 1) // _PAGE_SIZE
    page_shape = (pages + 1, _PAGE_SIZE, 1)
    latent_shape = (*page_shape, _LATENT)
    latent = torch.randn(latent_shape, generator=generator, device="cpu").to(
        torch.float8_e4m3fn
    )
    rope = (
        torch.randn((*page_shape, _ROPE), generator=generator, device="cpu") * 2
    ).to(dtype)
    scales = (
        torch.rand((*page_shape, _LATENT // _TILE), generator=generator, device="cpu")
        * 0.75
        + 0.25
    ) * 0.5
    # Match the SGLang DSA packed cache: 512 quantized bytes, 64 BF16 RoPE
    # values, then four FP32 scales per cached token.
    packed_bytes = torch.cat(
        (latent.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)),
        dim=-1,
    )
    packed = packed_bytes.npu().view(latent.dtype)
    page_table = torch.randperm(pages, generator=generator, device="cpu") + 1
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
        # DeepSeek-V3.2: QK width 128+64, YaRN factor=40, mscale_all_dim=1.
        192**-0.5 * (1 + 0.1 * math.log(40)) ** 2,
        2,
        2,
    )
    kwargs = dict(
        block_table=page_table.to(torch.int32).unsqueeze(0).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([context_length], dtype=torch.int32).npu(),
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
    [pytest.param(16, 4093, 2048, id="deepseek-v32-tp8-fp8-dsa-paged-decode")],
)
def test_kv_quant_sparse_flash_attention(heads, context_length, topk):
    import torch_npu
    from sgl_kernel_npu.operators.attention import (
        kv_quant_sparse_flash_attention_torch_npu,
    )
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH35:
        pytest.skip("The SGLang packed FP8 DSA cache path requires Ascend A5")
    if not callable(getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)):
        pytest.skip("Installed TorchNPU lacks npu_kv_quant_sparse_flash_attention")
    args, kwargs = generate_sparse_data(heads, context_length, topk)
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

    # BF16 NZ weights and the DeepSeek-V2 projection/absorption dimensions.
    dtype = torch.bfloat16
    token_x = torch.zeros((2, hidden), dtype=dtype)
    token_x[0, 0] = token_x[1, 1] = 1
    weight_dq = torch.zeros((hidden, cq_rank), dtype=dtype)
    weight_dkv_kr = torch.zeros((hidden, _LATENT + _ROPE), dtype=dtype)
    weight_dq[0, ::2] = 2
    weight_dq[1, 1::2] = -0.5
    weight_dkv_kr[0, ::2] = 2
    weight_dkv_kr[0, 1::2] = 0.25
    weight_dkv_kr[1, ::2] = -0.5
    weight_dkv_kr[1, 1::2] = 1
    weight_uq_qr = torch.zeros((cq_rank, heads * (head_dim + _ROPE)), dtype=dtype)
    weight_uq_qr[0, ::2] = 0.25
    weight_uq_qr[1, 1::2] = 0.5
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
    slots = [2 * _PAGE_SIZE + 4, 2 * _PAGE_SIZE - 1]
    kv_cache = torch.zeros((3, _PAGE_SIZE, 1, _LATENT), dtype=dtype)
    kr_cache = torch.zeros((3, _PAGE_SIZE, 1, _ROPE), dtype=dtype)
    args = (
        token_x.npu(),
        torch_npu.npu_format_cast(weight_dq.npu(), 29),
        torch_npu.npu_format_cast(weight_uq_qr.npu(), 29),
        weight_uk,
        torch_npu.npu_format_cast(weight_dkv_kr.npu(), 29),
        gamma_cq.npu(),
        gamma_ckv.npu(),
        sin.npu(),
        cos.npu(),
    )
    return args, dict(
        kv_cache=kv_cache.npu(),
        kr_cache=kr_cache.npu(),
        cache_index=torch.tensor(slots, dtype=torch.int64).npu(),
        rmsnorm_epsilon_cq=1e-6,
        rmsnorm_epsilon_ckv=1e-6,
        cache_mode="PA_BSND",
        query_norm_flag=True,
        weight_quant_mode=0,
    )


@pytest.mark.parametrize(
    "hidden, cq_rank, heads, head_dim",
    [pytest.param(5120, 1536, 16, 128, id="deepseek-v2-tp8-bf16-paged-prolog")],
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
        result_transform=lambda result: (result[0], result[1], result[3]),
        standard={"torch.bfloat16": AccuracyStandard(atol=1e-5, rtol=1.6e-2)},
    )
    check_performance(executor)
