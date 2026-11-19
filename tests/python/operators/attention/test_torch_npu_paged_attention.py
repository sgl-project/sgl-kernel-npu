"""Legacy ATB-backed attention through registered, caller-owned outputs."""

import pytest

torch = pytest.importorskip("torch")

_PAGE_SIZE = 128


@pytest.fixture
def require_legacy_atb():
    import torch_npu
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    # The documented ATB ND paged-cache layout is for Atlas A2/A3. The 950
    # build is not covered by these legacy interfaces' documented examples.
    # https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/acce/ascendtb/ascendtb_01_0202.html
    if get_hardware().generation is not Generation.ARCH22:
        pytest.skip("Legacy ATB paged attention test requires Atlas A2/A3")

    def require(symbol):
        if not callable(getattr(torch_npu, symbol, None)):
            pytest.skip(
                f"Installed TorchNPU does not export the legacy {symbol} binding"
            )

    return require


def _make_tensors(shapes):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return tuple(
        torch.randn(shape, generator=generator, device="cpu").to(torch.float16)
        for shape in shapes
    )


def _gather_cache(cache, physical_pages, length):
    return cache[physical_pages].flatten(0, 1)[:length].float()


def _dense_attention(query, key, value, scale, *, causal=False):
    # All inputs stay on CPU so the reference never executes the SDK backend.
    repeats = query.size(1) // key.size(1)
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    logits = torch.einsum("qhd,khd->hqk", query.float(), key) * scale
    if causal:
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool, device="cpu").triu(1)
        logits = logits.masked_fill(mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", logits.softmax(dim=-1), value).to(query.dtype)


@pytest.mark.parametrize(
    "q_heads,kv_heads,head_dim,tokens",
    [pytest.param(16, 4, 128, 128, id="qwen3-8b-tp2-paged-prefill")],
)
def test_flash_attention_qlens_causal_prefill(
    require_legacy_atb, q_heads, kv_heads, head_dim, tokens
):
    from sgl_kernel_npu.operators.attention import flash_attention_qlens

    require_legacy_atb("_npu_flash_attention_qlens")
    cache_shape = (2, _PAGE_SIZE, kv_heads, head_dim)
    query, key_cache, value_cache = _make_tensors(
        ((tokens, q_heads, head_dim), cache_shape, cache_shape)
    )
    # SGLang AscendAttnBackend.forward_extend flattens Q heads and passes
    # per-request query/context lengths on the host. FP16 uses an additive mask.
    mask = torch.full((tokens, tokens), float("-inf"), dtype=query.dtype).triu_(1)
    scale = head_dim**-0.5
    expected = _dense_attention(
        query,
        _gather_cache(key_cache, [1], tokens),
        _gather_cache(value_cache, [1], tokens),
        scale,
        causal=True,
    )
    query = query.flatten(1).npu()
    out = torch.full_like(query, float("nan"))
    lengths = torch.tensor([tokens], dtype=torch.int32, device="cpu")
    flash_attention_qlens(
        query=query,
        key_cache=key_cache.npu(),
        value_cache=value_cache.npu(),
        mask=mask,
        block_table=torch.tensor([[1]], dtype=torch.int32),
        seq_len=lengths,
        context_lens=lengths,
        scale_value=scale,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        out=out,
    )
    torch.testing.assert_close(out.cpu(), expected.flatten(1), atol=4e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "q_heads,kv_heads,head_dim,lengths",
    [pytest.param(16, 4, 128, (255, 129), id="qwen3-8b-tp2-paged-decode")],
)
def test_paged_attention_decode(
    require_legacy_atb, q_heads, kv_heads, head_dim, lengths
):
    from sgl_kernel_npu.operators.attention import paged_attention

    require_legacy_atb("_npu_paged_attention")
    cache_shape = (4, _PAGE_SIZE, kv_heads, head_dim)
    query, key_cache, value_cache = _make_tensors(
        ((len(lengths), q_heads, head_dim), cache_shape, cache_shape)
    )
    scale = head_dim**-0.5
    physical_pages = [[2, 0], [3, 1]]
    expected = torch.cat(
        [
            _dense_attention(
                query[i : i + 1],
                _gather_cache(key_cache, pages, length),
                _gather_cache(value_cache, pages, length),
                scale,
            )
            for i, (pages, length) in enumerate(
                zip(physical_pages, lengths, strict=True)
            )
        ]
    )
    query = query.npu()
    out = torch.full_like(query, float("nan"))
    paged_attention(
        query=query,
        key_cache=key_cache.npu(),
        value_cache=value_cache.npu(),
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        scale_value=scale,
        block_table=torch.tensor(physical_pages, dtype=torch.int32),
        context_lens=torch.tensor(lengths, dtype=torch.int32, device="cpu"),
        out=out,
        impl="torch_npu",
    )
    torch.testing.assert_close(out.cpu(), expected, atol=4e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "q_heads,latent_dim,rope_dim,length",
    [pytest.param(16, 512, 64, 255, id="deepseek-v3-tp8-mla-paged-decode")],
)
def test_paged_attention_mla_decode(
    require_legacy_atb, q_heads, latent_dim, rope_dim, length
):
    from sgl_kernel_npu.operators.attention import paged_attention_mla

    require_legacy_atb("_npu_paged_attention_mla")
    head_dim = latent_dim + rope_dim
    query, cache = _make_tensors(((1, q_heads, head_dim), (2, _PAGE_SIZE, 1, head_dim)))
    physical_pages = [1, 0]
    # The absorbed MLA query and cache concatenate the latent and rotary parts.
    # Only the latent portion of the cache supplies values.
    tokens = _gather_cache(cache, physical_pages, length)
    scale = (128 + rope_dim) ** -0.5
    expected = _dense_attention(query, tokens, tokens[..., :latent_dim], scale)
    out = torch.full((1, q_heads, latent_dim), float("nan"), dtype=query.dtype)
    paged_attention_mla(
        query=query.npu(),
        key_cache=cache.npu(),
        num_kv_heads=1,
        num_heads=q_heads,
        scale_value=scale,
        block_table=torch.tensor([physical_pages], dtype=torch.int32),
        context_lens=torch.tensor([length], dtype=torch.int32, device="cpu"),
        mla_vheadsize=latent_dim,
        out=out,
    )
    torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=6e-3)
