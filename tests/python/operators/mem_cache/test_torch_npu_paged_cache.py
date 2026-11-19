"""Check cache updates through the public TorchNPU operator interfaces."""

import pytest

torch = pytest.importorskip("torch")


def generate_paged_cache_inputs(slot_ids):
    slot_count = len(slot_ids)
    key = torch.randn(slot_count, 2, 128, dtype=torch.float16)
    value = torch.randn_like(key)
    key_cache = torch.zeros(2, 16, 2, 128, dtype=torch.float16)
    value_cache = torch.full_like(key_cache, -1)
    slots = torch.tensor(slot_ids, dtype=torch.int32)
    return key, value, key_cache, value_cache, slots


@pytest.mark.parametrize("name", ["reshape_and_cache", "scatter_pa_kv_cache"])
@pytest.mark.parametrize(
    "slot_ids",
    [(0,), (31, 15, 16)],
    ids=["single-token", "unordered-cross-page"],
)
def test_paged_cache_write(name, slot_ids):
    from sgl_kernel_npu.operators import mem_cache

    key, value, key_cache, value_cache, slots = generate_paged_cache_inputs(slot_ids)
    expected_key_cache = key_cache.cpu().clone()
    expected_value_cache = value_cache.cpu().clone()
    cpu_slots = slots.cpu().long()
    expected_key_cache.flatten(0, 1).index_copy_(0, cpu_slots, key.cpu())
    expected_value_cache.flatten(0, 1).index_copy_(0, cpu_slots, value.cpu())

    kwargs = {"cache_mode": "Norm"} if name == "scatter_pa_kv_cache" else {}
    operator = getattr(mem_cache, name)
    assert operator(key, value, key_cache, value_cache, slots, **kwargs) is None

    torch.testing.assert_close(key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value_cache, rtol=0, atol=0)


def test_kv_rmsnorm_rope_cache_paged():
    from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache

    # DeepSeek MLA latent (512) and rotary (64) dimensions with two decode tokens.
    tokens, page_size = 2, 16
    latent = torch.randn(tokens, 1, 1, 512, dtype=torch.float16)
    rope = torch.full((tokens, 1, 1, 64), 0.75, dtype=torch.float16)
    kv = torch.cat((latent, rope), dim=-1)
    gamma = torch.rand(512, dtype=torch.float16)
    cos = torch.ones(tokens, 1, 1, 64, dtype=torch.float16)
    sin = torch.zeros_like(cos)
    index = torch.tensor([page_size + 3, 0], dtype=torch.int64)
    key_cache = torch.zeros(2, page_size, 1, 64, dtype=torch.float16)
    ckv_cache = torch.zeros(2, page_size, 1, 512, dtype=torch.float16)

    latent_fp32 = latent.cpu().float()
    inverse_rms = (latent_fp32.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
    expected_norm = (latent_fp32 * inverse_rms * gamma.cpu().float()).to(kv.dtype)
    expected_key_cache = torch.zeros_like(key_cache, device="cpu")
    expected_ckv_cache = torch.zeros_like(ckv_cache, device="cpu")
    cpu_index = index.cpu()
    expected_key_cache.flatten(0, 1).index_copy_(0, cpu_index, rope.cpu().squeeze(1))
    expected_ckv_cache.flatten(0, 1).index_copy_(0, cpu_index, expected_norm.squeeze(1))

    output_key_cache, output_ckv_cache, output_rope, output_norm = (
        kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            index,
            key_cache,
            ckv_cache,
            cache_mode="PA_BNSD",
            is_output_kv=True,
        )
    )
    torch.testing.assert_close(
        output_key_cache.cpu(), expected_key_cache, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output_ckv_cache.cpu(), expected_ckv_cache, rtol=0.004, atol=0.003
    )
    torch.testing.assert_close(key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(
        ckv_cache.cpu(), expected_ckv_cache, rtol=0.004, atol=0.003
    )
    torch.testing.assert_close(output_rope.cpu(), rope.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(output_norm.cpu(), expected_norm, rtol=0.004, atol=0.003)
