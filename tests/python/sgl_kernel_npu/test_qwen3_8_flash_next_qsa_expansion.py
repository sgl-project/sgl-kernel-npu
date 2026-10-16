"""Model-contract expansion tests migrated from the frozen sandbox.

Legacy holes/partial-block/wrapping cases are outside the new GPU-like contract.
The reference below runs on CPU only; it is not a production fallback.
"""

import itertools

import pytest
import torch
import torch_npu
from sgl_kernel_npu.qwen3_8_flash_next import qsa_expansion as module
from sgl_kernel_npu.qwen3_8_flash_next.qsa_expansion import (
    can_run_block_expansion,
    expand_blocks,
)

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
)


def scalar_oracle(blocks, positions, lengths, ratio, topk):
    result = []
    for row, position, length in zip(
        blocks.tolist(), positions.tolist(), lengths.tolist()
    ):
        raw = []
        for block in row:
            raw.extend(
                [
                    block * ratio + offset if block >= 0 else -1
                    for offset in range(ratio)
                ]
            )
        raw = [v if 0 <= v < length else -1 for v in raw[:topk]]
        start = ((position + 1) // ratio) * ratio
        count = position + 1 - start
        raw += [
            start + j if j < count and start + j < length else -1
            for j in range(ratio - 1)
        ]
        raw = [v for v in raw if v >= 0] + [v for v in raw if v < 0]
        result.append(raw)
    return (
        torch.tensor(result, dtype=torch.int64)
        .to(torch.int32)
        .reshape(blocks.shape[0], topk + ratio - 1)
    )


def torch_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed block indices into fixed-width logical token indices."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat([expanded, tail], dim=1)
    # Keep all valid entries contiguous. This is required by the FA2 packing path.
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


reference = torch_expand_qsa_block_indices

PATTERNS = ("full", "short", "mixed", "padding", "graph_padding")


def make_model_case(
    rows,
    device="cpu",
    ratio=4,
    topk=2048,
    pattern="full",
    dtype=torch.int32,
    layout="contiguous",
    seed=0,
):
    if ratio < 2 or topk % ratio or topk // ratio not in (512, 2048):
        raise ValueError("model parameters require ratio>=2 and block_topk=512/2048")
    if pattern not in PATTERNS:
        raise ValueError("pattern is outside the model contract")
    k = topk // ratio
    row = torch.arange(rows, dtype=torch.int64)
    available = torch.full((rows,), k + 19, dtype=torch.int64)
    remainder = (row + seed) % ratio
    if pattern == "short":
        available.fill_(3)
    elif pattern == "mixed":
        available = (row * 37 + seed) % (k + 23)
    elif pattern in ("padding", "graph_padding"):
        available.zero_()
        remainder.fill_(0 if pattern == "padding" else 1)
    lengths = available * ratio + remainder
    positions = lengths - 1
    columns = torch.arange(k).unsqueeze(0)
    blocks = (columns * 17 + row[:, None] * 3 + seed) % available.clamp_min(1)[:, None]
    blocks = torch.where(columns < available[:, None], blocks, -1).to(dtype)
    tensors = [blocks, positions, lengths.to(torch.int32)]
    if layout == "strided":
        tensors = [torch.stack((x, x), -1).to(device)[..., 0] for x in tensors]
    elif layout == "broadcast" and rows:
        tensors = [x[:1].to(device).expand((rows, *x.shape[1:])) for x in tensors]
    else:
        tensors = [x.to(device) for x in tensors]
    return (*tensors, ratio, topk)


def expected(args):
    cpu = tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in args)
    if cpu[0].shape[0] <= 129:
        return scalar_oracle(*cpu)
    # Full exact comparison, chunked only to bound host oracle workspace.
    chunks = [
        reference(*(t[start : start + 128] for t in cpu[:3]), *cpu[3:])
        for start in range(0, cpu[0].shape[0], 128)
    ]
    return torch.cat(chunks)


def assert_preconditions(blocks, positions, lengths, ratio):
    """Test-only host checks; never called by the production wrapper."""
    b, p, l = (x.cpu().long() for x in (blocks, positions, lengths))
    valid = b >= 0
    count = valid.sum(1)
    assert torch.equal(valid, torch.arange(b.shape[1])[None, :] < count[:, None])
    assert torch.all(b[~valid] == -1)
    assert torch.all(
        ((b + 1) * ratio)[valid] <= torch.minimum(p + 1, l)[:, None].expand_as(b)[valid]
    )
    assert torch.all(l >= 0) and torch.all(l <= 2**31 - 1)
    assert torch.all(p >= -1) and torch.all(p < 2**31 - 1)


@pytest.mark.parametrize(
    "rows", [0, 1, 4, 16, 64, 127, 128, 129, 512, 4095, 4096, 4097, 8192, 16384]
)
@pytest.mark.parametrize(
    "pattern", ["full", "short", "mixed", "padding", "graph_padding"]
)
def test_model_shapes(rows, pattern):
    args = make_model_case(rows, device="npu", pattern=pattern)
    before = [x.clone() for x in args[:3]]
    actual = expand_blocks(*args)
    assert (
        actual.shape == (rows, 2051)
        and actual.dtype == torch.int32
        and actual.is_contiguous()
    )
    actual_cpu, expected_cpu = actual.cpu(), expected(args)
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=0, rtol=0)
    for x, saved in zip(args[:3], before):
        assert torch.equal(x, saved)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("ratio", [2, 3, 4, 5, 8, 16, 33, 257])
@pytest.mark.parametrize("path", ["triton", "native"])
def test_parameterized_ratio_and_tail_remainders(k, ratio, path, monkeypatch):
    monkeypatch.setattr(
        module, "_NATIVE_MIN_ROWS", 0 if path == "native" else 2**63 - 1
    )
    # Every remainder; complete blocks and a full padded suffix are both present.
    args = make_model_case(
        ratio, device="npu", ratio=ratio, topk=k * ratio, pattern="short"
    )
    assert_preconditions(*args[:3], ratio)
    torch.testing.assert_close(
        expand_blocks(*args).cpu(), expected(args), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "dtypes", list(itertools.product([torch.int32, torch.int64], repeat=3))
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "broadcast"])
@pytest.mark.parametrize("path", ["triton", "native"])
def test_dtype_and_layout(dtypes, layout, path, monkeypatch):
    monkeypatch.setattr(
        module, "_NATIVE_MIN_ROWS", 0 if path == "native" else 2**63 - 1
    )
    args = list(
        make_model_case(7, device="npu", pattern="mixed", seed=3, layout=layout)
    )
    # Preserve strides after conversion by slicing duplicated storage again.
    args[:3] = [
        torch.stack((t.to(d), t.to(d)), -1)[..., 0] if layout == "strided" else t.to(d)
        for t, d in zip(args[:3], dtypes)
    ]
    torch.testing.assert_close(
        expand_blocks(*args).cpu(), expected(args), atol=0, rtol=0
    )


@pytest.mark.parametrize("path", ["triton", "native"])
@pytest.mark.parametrize("ratio", [2, 3, 4, 5, 8, 257])
def test_order_duplicates_and_int32_end(path, ratio, monkeypatch):
    monkeypatch.setattr(
        module, "_NATIVE_MIN_ROWS", 0 if path == "native" else 2**63 - 1
    )
    args = list(
        make_model_case(
            3,
            device="npu",
            dtype=torch.int64,
            pattern="padding",
            ratio=ratio,
            topk=512 * ratio,
        )
    )
    last_full_block = (2**31 - 1) // ratio - 1
    args[0][:, :3] = torch.tensor([last_full_block, 1, last_full_block], device="npu")
    args[1].fill_(2**31 - 2)
    args[2].fill_(2**31 - 1)
    assert_preconditions(*args[:3], ratio)
    torch.testing.assert_close(
        expand_blocks(*args).cpu(), expected(args), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "kind",
    [
        "budget",
        "ratio",
        "k",
        "bool",
        "float_parameter",
        "rank",
        "vector_shape",
        "dtype",
        "cpu",
        "width_overflow",
    ],
)
def test_reject_metadata(kind):
    args = list(make_model_case(2, device="npu"))
    if kind == "budget":
        args[4] += 1
    elif kind == "ratio":
        args[3] = 1
    elif kind == "k":
        args[0] = args[0][:, :256]
        args[4] = 1024
    elif kind == "bool":
        args[3] = True
    elif kind == "float_parameter":
        args[4] = 2048.0
    elif kind == "rank":
        args[0] = args[0].flatten()
    elif kind == "vector_shape":
        args[1] = args[1].reshape(1, 2)
    elif kind == "dtype":
        args[0] = args[0].float()
    elif kind == "cpu":
        args[2] = args[2].cpu()
    elif kind == "width_overflow":
        args[3] = 2**23
        args[4] = 512 * args[3]
    assert not can_run_block_expansion(*args)
    with pytest.raises(ValueError, match="metadata"):
        expand_blocks(*args)


@pytest.mark.parametrize("rows", [65534, 65535, 65536, 65537])
def test_bounded_grid_loop(rows, monkeypatch):
    # Explicitly exercise Triton, regardless of the default large-R policy.
    monkeypatch.setattr(module, "_NATIVE_MIN_ROWS", 2**63 - 1)
    args = make_model_case(
        rows,
        device="npu",
        ratio=2,
        topk=1024,
        pattern="mixed",
        layout="broadcast",
        seed=7,
    )
    actual = expand_blocks(*args)
    one = expected(tuple(t[:1] if isinstance(t, torch.Tensor) else t for t in args))
    for start in range(0, rows, 1024):
        chunk = actual[start : start + 1024].cpu()
        torch.testing.assert_close(
            chunk, one.expand(chunk.shape[0], -1), atol=0, rtol=0
        )


@pytest.mark.parametrize("path", ["triton", "native"])
def test_int64_row_and_column_addresses(path, monkeypatch):
    monkeypatch.setattr(
        module, "_NATIVE_MIN_ROWS", 0 if path == "native" else 2**63 - 1
    )
    # Cross signed-int32 ELEMENT offsets, touching only the two selected rows.
    for column_stride in (False, True):
        stride = 2**31 + 64
        if column_stride:
            # Disjoint rows, strided columns: last column crosses the signed
            # int32 element offset while allocation stays near 8 GiB.
            shape, strides = (2, 512), (1, (2**31 // 511 + 1))
        else:
            shape, strides = (2, 512), (stride, 1)
        size = sum((n - 1) * s for n, s in zip(shape, strides)) + 1
        storage = torch.empty(size, dtype=torch.int32, device="npu")
        blocks = storage.as_strided(shape, strides)
        base = make_model_case(2, device="npu")
        blocks.copy_(base[0])
        args = (blocks, *base[1:])
        torch.testing.assert_close(
            expand_blocks(*args).cpu(), expected(base), atol=0, rtol=0
        )
        del blocks, storage, args


@pytest.mark.parametrize(
    "rows,k,ratio",
    [
        (1, 512, 4),
        (127, 512, 4),
        (128, 512, 4),
        (129, 512, 3),
        (4097, 512, 4),
        (8192, 2048, 4),
        (17, 2048, 8),
    ],
)
@pytest.mark.parametrize("path", ["default", "triton", "native"])
def test_graph_input_updates(rows, k, ratio, path, monkeypatch):
    if path != "default":
        monkeypatch.setattr(
            module, "_NATIVE_MIN_ROWS", 0 if path == "native" else 2**63 - 1
        )
    args = make_model_case(rows, device="npu", topk=k * ratio, ratio=ratio)
    for _ in range(3):
        expand_blocks(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = expand_blocks(*args)
    for pattern in ["mixed", "short", "padding", "graph_padding", "full"]:
        replacement = make_model_case(
            rows, device="npu", topk=k * ratio, ratio=ratio, pattern=pattern, seed=1
        )
        for old, new in zip(args[:3], replacement[:3]):
            old.copy_(new)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected(replacement), atol=0, rtol=0)


def test_native_chunk_graph(monkeypatch):
    monkeypatch.setattr(module, "_NATIVE_MIN_ROWS", 0)
    monkeypatch.setattr(module, "_NATIVE_CHUNK_ELEMENTS", 31 * 2051)
    args = make_model_case(129, device="npu", pattern="full")
    for _ in range(3):
        expand_blocks(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = expand_blocks(*args)
    for pattern in ["mixed", "padding", "graph_padding", "short"]:
        replacement = make_model_case(129, device="npu", pattern=pattern, seed=2)
        for a, b in zip(args[:3], replacement[:3]):
            a.copy_(b)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected(replacement), atol=0, rtol=0)
