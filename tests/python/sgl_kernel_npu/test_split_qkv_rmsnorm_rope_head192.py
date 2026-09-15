import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
from sgl_kernel_npu.utils.triton_utils import get_device_properties

# Keep the existing operator test's absolute tolerance. This is not an
# end-to-end model-quality or performance acceptance threshold.
ATOL = 5e-2
CACHE_DTYPES = (torch.float32, torch.bfloat16)


def _make_case(
    tokens,
    q_heads,
    *,
    kv_heads=None,
    head_dim=192,
    rope_dim=None,
    input_dtype=torch.bfloat16,
    weight_dtype=None,
    cache_dtype=torch.float32,
    cos_cache_dtype=None,
    eps=1e-5,
    bias=False,
    is_neox_style=True,
    seed=17,
    scale=1.0,
    position_shift=0,
):
    """Build independent CPU inputs in the public host function's layout."""
    kv_heads = q_heads if kv_heads is None else kv_heads
    rope_dim = head_dim if rope_dim is None else rope_dim
    weight_dtype = input_dtype if weight_dtype is None else weight_dtype
    cos_cache_dtype = cache_dtype if cos_cache_dtype is None else cos_cache_dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    qkv = (torch.randn(tokens, q_width + 2 * kv_width, generator=generator) * scale).to(
        input_dtype
    )
    q_weight = (0.5 + torch.rand(head_dim, generator=generator)).to(weight_dtype)
    k_weight = (0.5 + torch.rand(head_dim, generator=generator)).to(weight_dtype)
    q_bias = torch.randn(head_dim, generator=generator).to(weight_dtype)
    k_bias = torch.randn(head_dim, generator=generator).to(weight_dtype)
    positions = torch.tensor([0, 1, 7, 127, 4096, 32767, 131071])
    positions = positions[torch.arange(tokens) % len(positions)] + position_shift
    inv_frequency = 1.0 / (
        8000000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim)
    )
    angles = torch.outer(positions.float(), inv_frequency)
    angles = torch.cat((angles, angles), dim=-1)
    return {
        "input": qkv,
        "sin": angles.sin().to(cache_dtype).reshape(tokens, 1, 1, rope_dim),
        "cos": angles.cos().to(cos_cache_dtype).reshape(tokens, 1, 1, rope_dim),
        "q_hidden_size": q_width,
        "kv_hidden_size": kv_width,
        "head_dim": head_dim,
        "eps": eps,
        "q_weight": q_weight if eps is not None else None,
        "k_weight": k_weight if eps is not None else None,
        "q_bias": q_bias if bias else None,
        "k_bias": k_bias if bias else None,
        "is_neox_style": is_neox_style,
    }


def _to_device(case):
    device = torch.device("npu", torch.npu.current_device())
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }


def _reference(case):
    """FP32 norm/RoPE, preserving the input/cache quantization at the boundary."""
    tokens, dim = case["input"].shape[0], case["head_dim"]
    q_width, kv_width = case["q_hidden_size"], case["kv_hidden_size"]
    q, k, v = case["input"].cpu().split((q_width, kv_width, kv_width), dim=-1)
    rope_dim = case["sin"].shape[-1]
    sin = case["sin"].cpu().float().reshape(tokens, 1, rope_dim)
    cos = case["cos"].cpu().float().reshape(tokens, 1, rope_dim)
    result = []
    for name, values, width in (("q", q, q_width), ("k", k, kv_width)):
        values = values.float().reshape(tokens, width // dim, dim)
        if case["eps"] is not None:
            values = values / torch.sqrt(
                values.square().mean(dim=-1, keepdim=True) + case["eps"]
            )
            values = values * case[name + "_weight"].cpu().float().reshape(dim)
            if case[name + "_bias"] is not None:
                values = values + case[name + "_bias"].cpu().float().reshape(dim)
        rot = values[..., :rope_dim]
        half = rope_dim // 2
        if case["is_neox_style"]:
            left, right = rot[..., :half], rot[..., half:]
            roped = torch.cat(
                (
                    left * cos[..., :half] - right * sin[..., :half],
                    right * cos[..., half:] + left * sin[..., half:],
                ),
                dim=-1,
            )
        else:
            # The host receives duplicated-half caches. The kernel takes their
            # first half and repeats each frequency for an adjacent element pair.
            even, odd = rot[..., 0::2], rot[..., 1::2]
            pair_sin, pair_cos = sin[..., :half], cos[..., :half]
            roped = torch.stack(
                (even * pair_cos - odd * pair_sin, odd * pair_cos + even * pair_sin),
                dim=-1,
            ).flatten(start_dim=-2)
        output = torch.cat((roped, values[..., rope_dim:]), dim=-1)
        result.append(output.reshape(tokens, width))
    return (*result, v.contiguous())


def _assert_outputs(outputs, expected):
    for actual, wanted in zip(outputs[:2], expected[:2]):
        assert actual.dtype == expected[2].dtype
        actual = actual.detach().float().cpu()
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, wanted.float().cpu(), rtol=0, atol=ATOL)
    actual_v, expected_v = outputs[2].detach().cpu(), expected[2].cpu()
    assert actual_v.dtype == expected_v.dtype
    assert torch.equal(
        actual_v.contiguous().view(torch.uint8),
        expected_v.contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=("fp32", "bf16"))
@pytest.mark.parametrize(
    "tokens,heads",
    [(1, 1), (8, 2), (33, 4), (8, 8), (1, 16), (8, 32), (1, 64), (0, 4)],
)
def test_head192_public_host_shapes(tokens, heads, cache_dtype):
    # Local head counts exercise valid tensor partitions, not TP communication.
    case = _make_case(tokens, heads, cache_dtype=cache_dtype, seed=tokens + heads)
    outputs = split_qkv_rmsnorm_rope(**_to_device(case))
    torch.npu.synchronize()
    _assert_outputs(outputs, _reference(case))


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=("fp32", "bf16"))
def test_head192_public_host_crosses_row_step(cache_dtype):
    _, vector_cores = get_device_properties()
    tokens = (vector_cores + 3) // 4 + 1
    case = _make_case(tokens, 4, cache_dtype=cache_dtype)
    outputs = split_qkv_rmsnorm_rope(**_to_device(case))
    torch.npu.synchronize()
    _assert_outputs(outputs, _reference(case))


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=("fp32", "bf16"))
def test_head192_public_host_2d_cache(cache_dtype):
    case = _make_case(8, 4, cache_dtype=cache_dtype)
    for key in ("sin", "cos"):
        case[key] = case[key].reshape(8, 192)
    outputs = split_qkv_rmsnorm_rope(**_to_device(case))
    torch.npu.synchronize()
    _assert_outputs(outputs, _reference(case))


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=("fp32", "bf16"))
@pytest.mark.parametrize("eps,scale", [(1e-6, 1.0), (1e-5, 0.0), (1e-4, 1e-4)])
def test_head192_public_host_norm_boundaries(cache_dtype, eps, scale):
    case = _make_case(8, 4, cache_dtype=cache_dtype, eps=eps, scale=scale)
    outputs = split_qkv_rmsnorm_rope(**_to_device(case))
    torch.npu.synchronize()
    _assert_outputs(outputs, _reference(case))


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=("fp32", "bf16"))
def test_head192_public_host_graph_changed_input(cache_dtype):
    case = _to_device(_make_case(8, 4, cache_dtype=cache_dtype))
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        for _ in range(3):
            split_qkv_rmsnorm_rope(**case)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    # Capture the public host, including its output allocations. The captured
    # outputs are retained; subsequent replay updates contents, not addresses.
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        outputs = split_qkv_rmsnorm_rope(**case)
    torch.npu.synchronize()
    addresses = tuple(output.data_ptr() for output in outputs)
    for seed, shift in ((73, 13), (101, 257)):
        changed = _make_case(
            8, 4, cache_dtype=cache_dtype, seed=seed, position_shift=shift
        )
        with torch.npu.stream(stream):
            for key in ("input", "sin", "cos", "q_weight", "k_weight"):
                case[key].copy_(changed[key])
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        assert tuple(output.data_ptr() for output in outputs) == addresses
        expected = _reference(changed)
        _assert_outputs(outputs, expected)
        with torch.npu.stream(stream):
            eager_outputs = split_qkv_rmsnorm_rope(**case)
        stream.synchronize()
        _assert_outputs(eager_outputs, expected)
        _assert_outputs(outputs, eager_outputs)


@pytest.mark.parametrize(
    "head_dim,q_heads,kv_heads,rope_dim,eps,bias",
    [
        (64, 8, 2, 64, 1e-6, False),
        (128, 12, 2, 64, 1e-6, True),
        (256, 4, 1, 256, None, False),
    ],
)
def test_existing_head_dimensions_public_host(
    head_dim, q_heads, kv_heads, rope_dim, eps, bias
):
    # Exercise existing modes while adding a new supported head dimension.
    # The original operator test module remains part of the regression run.
    case = _make_case(
        13,
        q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        cache_dtype=torch.bfloat16,
        eps=eps,
        bias=bias,
    )
    outputs = split_qkv_rmsnorm_rope(**_to_device(case))
    torch.npu.synchronize()
    _assert_outputs(outputs, _reference(case))


@pytest.mark.parametrize(
    "options,equivalent_views",
    [
        pytest.param({"q_heads": 4, "kv_heads": 2}, False, id="gqa_ratio2"),
        pytest.param({"q_heads": 6, "kv_heads": 2}, False, id="gqa_ratio3"),
        pytest.param({"eps": None}, False, id="no_norm_or_weights"),
        pytest.param({"bias": True}, False, id="paired_norm_bias"),
        pytest.param({"rope_dim": 128}, False, id="partial_rope128"),
        pytest.param({"rope_dim": 96}, False, id="partial_rope96"),
        pytest.param({"is_neox_style": False}, False, id="non_neox_full"),
        pytest.param(
            {"is_neox_style": False, "rope_dim": 96},
            False,
            id="non_neox_partial96",
        ),
        pytest.param(
            {"input_dtype": torch.float16, "cache_dtype": torch.float16},
            False,
            id="fp16",
        ),
        pytest.param({"input_dtype": torch.float32}, False, id="fp32"),
        pytest.param(
            {"weight_dtype": torch.float32, "cos_cache_dtype": torch.bfloat16},
            False,
            id="fp32_weights_mixed_cache",
        ),
        pytest.param({}, True, id="contiguous_equivalent_views"),
    ],
)
def test_head192_public_host_existing_modes(options, equivalent_views):
    # Cover distinct existing branches without multiplying every configuration.
    parameters = {"tokens": 8, "q_heads": 4, **options}
    case = _make_case(**parameters)
    if equivalent_views:
        # These views keep the same contiguous flat data and the cache's last
        # dimension. The kernel does not consume the other shape dimensions.
        case["q_weight"] = case["q_weight"].reshape(1, 192)
        case["k_weight"] = case["k_weight"].reshape(192, 1)
        case["sin"] = case["sin"].reshape(2, 4, 192)
        case["cos"] = case["cos"].reshape(8, 192)
    outputs = split_qkv_rmsnorm_rope(**_to_device(case))
    torch.npu.synchronize()
    _assert_outputs(outputs, _reference(case))
