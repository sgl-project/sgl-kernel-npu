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
    cache_dtype=torch.float32,
    eps=1e-5,
    bias=False,
    seed=17,
    scale=1.0,
    position_shift=0,
):
    """Build independent CPU inputs in the public host function's layout."""
    kv_heads = q_heads if kv_heads is None else kv_heads
    rope_dim = head_dim if rope_dim is None else rope_dim
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    qkv = (torch.randn(tokens, q_width + 2 * kv_width, generator=generator) * scale).to(
        torch.bfloat16
    )
    q_weight = (0.5 + torch.rand(head_dim, generator=generator)).to(torch.bfloat16)
    k_weight = (0.5 + torch.rand(head_dim, generator=generator)).to(torch.bfloat16)
    q_bias = torch.randn(head_dim, generator=generator).to(torch.bfloat16)
    k_bias = torch.randn(head_dim, generator=generator).to(torch.bfloat16)
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
        "cos": angles.cos().to(cache_dtype).reshape(tokens, 1, 1, rope_dim),
        "q_hidden_size": q_width,
        "kv_hidden_size": kv_width,
        "head_dim": head_dim,
        "eps": eps,
        "q_weight": q_weight if eps is not None else None,
        "k_weight": k_weight if eps is not None else None,
        "q_bias": q_bias if bias else None,
        "k_bias": k_bias if bias else None,
        "is_neox_style": True,
    }


def _to_device(case):
    device = torch.device("npu", torch.npu.current_device())
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }


def _reference(case):
    """FP32 RMSNorm and NeoX RoPE with no intermediate BF16 rounding."""
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
            values = values * case[name + "_weight"].cpu().float()
            if case[name + "_bias"] is not None:
                values = values + case[name + "_bias"].cpu().float()
        rot = values[..., :rope_dim]
        half = rope_dim // 2
        rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
        output = torch.cat((rot * cos + rotated * sin, values[..., rope_dim:]), dim=-1)
        result.append(output.reshape(tokens, width))
    return (*result, v.contiguous())


def _assert_outputs(outputs, expected):
    for actual, wanted in zip(outputs[:2], expected[:2]):
        actual = actual.detach().float().cpu()
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, wanted.float().cpu(), rtol=0, atol=ATOL)
    actual_v, expected_v = outputs[2].detach().cpu(), expected[2].cpu()
    assert actual_v.dtype == expected_v.dtype == torch.bfloat16
    assert torch.equal(
        actual_v.contiguous().view(torch.int16),
        expected_v.contiguous().view(torch.int16),
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
    # These do not inherit the new 192-only Q=KV/full-RoPE/no-bias requirements.
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


@pytest.mark.parametrize("eps", [None, 0.0, -1e-5, float("inf"), float("nan")])
def test_head192_rejects_unsupported_eps(eps):
    case = _to_device(_make_case(4, 4))
    case["eps"] = eps
    with pytest.raises(ValueError):
        split_qkv_rmsnorm_rope(**case)


def test_head192_rejects_noninteger_width():
    case = _to_device(_make_case(4, 4))
    case["q_hidden_size"] = float(case["q_hidden_size"])
    case["kv_hidden_size"] = float(case["kv_hidden_size"])
    with pytest.raises((ValueError, TypeError)):
        split_qkv_rmsnorm_rope(**case)


@pytest.mark.parametrize(
    "invalid",
    [
        "input_dtype",
        "input_rank",
        "input_stride",
        "input_width",
        "zero_heads",
        "unequal_heads",
        "fractional_head",
        "q_weight_missing",
        "k_weight_missing",
        "weight_dtype",
        "weight_rank",
        "weight_length",
        "weight_stride",
        "q_bias",
        "k_bias",
        "non_neox",
        "partial_rope",
        "cache_dtype",
        "cache_dtype_mismatch",
        "cache_shape_mismatch",
        "cache_rank",
        "cache_first_dim",
        "cache_last_dim",
        "cache_numel",
        "cache_stride",
    ],
)
def test_head192_rejects_unsupported_layouts(invalid):
    case = _to_device(_make_case(4, 4))
    if invalid == "input_dtype":
        case["input"] = case["input"].float()
    elif invalid == "input_rank":
        case["input"] = case["input"].unsqueeze(0)
    elif invalid == "input_stride":
        case["input"] = case["input"].t().contiguous().t()
    elif invalid == "input_width":
        case["input"] = case["input"][:, :-1].contiguous()
    elif invalid == "zero_heads":
        case.update(q_hidden_size=0, kv_hidden_size=0)
        case["input"] = case["input"][:, :0]
    elif invalid == "unequal_heads":
        case = _to_device(_make_case(4, 8, kv_heads=4))
    elif invalid == "fractional_head":
        case.update(q_hidden_size=193, kv_hidden_size=193)
        case["input"] = case["input"][:, : 193 * 3].contiguous()
    elif invalid.endswith("_weight_missing"):
        case[invalid.removesuffix("_missing")] = None
    elif invalid == "weight_dtype":
        case["q_weight"] = case["q_weight"].float()
    elif invalid == "weight_rank":
        case["q_weight"] = case["q_weight"].unsqueeze(0)
    elif invalid == "weight_length":
        case["q_weight"] = case["q_weight"][:-1]
    elif invalid == "weight_stride":
        case["q_weight"] = torch.stack((case["q_weight"], case["q_weight"]), dim=1)[
            :, 0
        ]
    elif invalid in ("q_bias", "k_bias"):
        case[invalid] = torch.zeros_like(case["q_weight"])
    elif invalid == "non_neox":
        case["is_neox_style"] = False
    elif invalid == "partial_rope":
        for key in ("sin", "cos"):
            case[key] = case[key][..., :96].contiguous()
    elif invalid == "cache_dtype":
        for key in ("sin", "cos"):
            case[key] = case[key].half()
    elif invalid == "cache_dtype_mismatch":
        case["sin"] = case["sin"].to(torch.bfloat16)
    elif invalid == "cache_shape_mismatch":
        case["sin"] = case["sin"].reshape(4, 192)
    elif invalid == "cache_rank":
        for key in ("sin", "cos"):
            case[key] = case[key].flatten()
    elif invalid == "cache_first_dim":
        for key in ("sin", "cos"):
            case[key] = case[key].reshape(2, 2, 192)
    elif invalid == "cache_last_dim":
        for key in ("sin", "cos"):
            case[key] = case[key].reshape(4, 192, 1)
    elif invalid == "cache_numel":
        for key in ("sin", "cos"):
            case[key] = case[key].repeat(1, 1, 2, 1)
    elif invalid == "cache_stride":
        for key in ("sin", "cos"):
            case[key] = case[key].reshape(4, 192).t().contiguous().t()
    else:
        raise AssertionError("unhandled invalid case")
    with pytest.raises(ValueError):
        split_qkv_rmsnorm_rope(**case)


@pytest.mark.parametrize("field", ["input", "sin", "cos", "q_weight", "k_weight"])
def test_head192_rejects_cpu_tensor(field):
    case = _to_device(_make_case(4, 4))
    case[field] = case[field].cpu()
    with pytest.raises(ValueError):
        split_qkv_rmsnorm_rope(**case)
