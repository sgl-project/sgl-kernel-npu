"""Ascend 950 FP8 integration checks for ACLNN and legacy direct-launch wheels."""

import importlib.util

import pytest
import torch
import torch_npu

_original = torch_npu.npu_quant_lightning_indexer
import sgl_kernel_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed


def generate_data(batch=1, kv=8192, heads=16, page=128, qlen=1, topk=2048, offset=0):
    g = torch.Generator().manual_seed(20261029)
    per_request = (kv + page - 1) // page
    pages = batch * per_request

    def fp8(shape):
        raw = torch.randint(16, 96, shape, generator=g, device="cpu", dtype=torch.uint8)
        raw |= (
            torch.randint(0, 2, shape, generator=g, device="cpu", dtype=torch.uint8)
            * 128
        )
        return raw.view(torch.float8_e4m3fn)

    q = fp8((batch * qlen, heads, 128))
    k = fp8((pages, page, 1, 128))
    scale = torch.rand((pages, page, 1), generator=g, device="cpu") * 0.1 + 0.01
    w = torch.rand((batch * qlen, heads), generator=g, device="cpu").to(torch.bfloat16)
    qs = torch.rand((batch * qlen, heads), generator=g, device="cpu") * 0.1 + 0.01
    sq = torch.arange(1, batch + 1, dtype=torch.int32, device="cpu") * qlen
    sk = torch.tensor(
        [max(topk, kv - i * 17) for i in range(batch)], dtype=torch.int32, device="cpu"
    )
    table = (
        torch.randperm(pages, generator=g, device="cpu")
        .reshape(batch, per_request)
        .int()
    )
    packed = torch.empty((pages, page * 132), dtype=torch.uint8, device="cpu")
    packed[:, : page * 128] = k.view(torch.uint8).reshape(pages, page * 128)
    packed[:, page * 128 :] = scale.view(torch.uint8).reshape(pages, page * 4)
    q, k, scale, w, qs, sq, sk, table, packed = [
        x.npu() for x in (q, k, scale, w, qs, sq, sk, table, packed)
    ]
    if offset:
        backing = torch.empty(
            packed.numel() + offset, dtype=torch.uint8, device=q.device
        )
        view = backing[offset:].view_as(packed)
        view.copy_(packed)
        packed = view
    return [q, packed, w, qs, sq, sk, table], k, scale


def compare(args, k, scale, topk, sparse_mode=3):
    actual = quant_lightning_indexer_packed(*args, topk, sparse_mode)
    q, _, w, qs, sq, sk, table = args
    expected = torch.ops.npu.npu_quant_lightning_indexer(
        q,
        k,
        w,
        qs,
        scale,
        0,
        0,
        actual_seq_lengths_query=sq,
        actual_seq_lengths_key=sk,
        block_table=table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=sparse_mode,
    )
    if isinstance(expected, (tuple, list)):
        expected = expected[0]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        {},
        dict(batch=2, kv=511, topk=256),
        dict(batch=8, kv=1024, heads=64, topk=256),
        dict(page=16, kv=257, topk=256),
        dict(page=1024, kv=2049, topk=2048),
        dict(qlen=4),
        dict(offset=4),
    ],
)
def test_exact_topk(shape):
    args, k, scale = generate_data(**shape)
    compare(args, k, scale, shape.get("topk", 2048))


def test_nondefault_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        args, k, scale = generate_data(kv=511, topk=256)
        compare(args, k, scale, 256)


def test_graph_replay_preserves_tiling_for_different_page_sizes():
    """Each captured graph retains its own host-generated tiling parameters."""
    stream = torch.npu.Stream()
    cases = [
        generate_data(batch=1, kv=257, page=16, qlen=1, topk=128),
        generate_data(batch=2, kv=511, page=128, qlen=3, topk=128),
    ]
    for args, k, scale in cases:
        compare(args, k, scale, 128)

    graphs, outputs, expected = [], [], []
    with torch.npu.stream(stream):
        for args, _, _ in cases:
            expected.append(quant_lightning_indexer_packed(*args, 128, 3).clone())
    stream.synchronize()

    for args, _, _ in cases:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = quant_lightning_indexer_packed(*args, 128, 3)
        graphs.append(graph)
        outputs.append(output)

    for index in (1, 0, 1, 0):
        graphs[index].replay()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[index].cpu(), expected[index].cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        "cache_dtype",
        "weight_dtype",
        "scale_dtype",
        "seq_dtype",
        "offset",
        "topk",
        "cpu_query",
        "noncontiguous",
    ],
)
def test_rejected(bad):
    args, _, _ = generate_data(kv=256, topk=256, offset=1 if bad == "offset" else 0)
    index_dtype = {
        "cache_dtype": (1, torch.int8),
        "weight_dtype": (2, torch.float32),
        "scale_dtype": (3, torch.float16),
        "seq_dtype": (4, torch.int64),
    }
    if bad in index_dtype:
        index, dtype = index_dtype[bad]
        args[index] = args[index].to(dtype)
    if bad == "cpu_query":
        args[0] = args[0].cpu()
    if bad == "noncontiguous":
        args[0] = args[0][:, ::2, :]
    with pytest.raises(RuntimeError):
        quant_lightning_indexer_packed(*args, 0 if bad == "topk" else 256, 3)


def test_empty_and_fake():
    args, _, _ = generate_data(kv=256, topk=256)
    for index in (0, 2, 3):
        args[index] = args[index][:0]
    assert quant_lightning_indexer_packed(*args, 256, 3).shape == (0, 1, 256)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(x) for x in args]
        result = torch.ops.npu.sgl_quant_lightning_indexer_packed(*fake, 256, 3)
        assert result.shape == (0, 1, 256) and result.dtype == torch.int32


def test_existing_registration():
    assert torch_npu.npu_quant_lightning_indexer is _original
    assert importlib.util.find_spec("qli_packed") is None


def test_packaged_native_library():
    from pathlib import Path

    args, k, scale = generate_data(kv=256, topk=256)
    compare(args, k, scale, 256)
    maps = Path("/proc/self/maps").read_text()
    assert "libsgl_kernel_npu.so" in maps
    package = Path(sgl_kernel_npu.__file__).parent
    if (package / "_cann").is_dir():
        assert "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so" in maps
    else:
        # Legacy wheel must retain the NEXTN dependency alongside packed QLI.
        assert hasattr(torch.ops.npu, "build_tree_kernel_efficient")
        assert "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so" not in maps
    assert "sgl_qli_packed/op_api/lib/libcust_opapi.so" not in maps


@pytest.mark.parametrize("heads", [8, 24, 32])
@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_head_and_mask_specializations(heads, sparse_mode):
    args, k, scale = generate_data(batch=2, qlen=3, kv=511, heads=heads, topk=128)
    compare(args, k, scale, 128, sparse_mode)


def test_unsupported_head_count():
    args, _, _ = generate_data(kv=256, heads=12, topk=128)
    with pytest.raises(RuntimeError, match="head count"):
        quant_lightning_indexer_packed(*args, topk=128)
