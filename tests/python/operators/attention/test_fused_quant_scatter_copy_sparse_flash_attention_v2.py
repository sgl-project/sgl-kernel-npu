import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from sgl_kernel_npu.operators.attention.fused_quant_scatter_copy_sparse_flash_attention_v2 import (
    fused_quant_scatter_copy_sparse_flash_attention_v2,
)


def test_copy_sfa_schedule(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the host scheduling check")
    root = Path(__file__).resolve().parents[4]
    include = (
        root
        / "csrc/attention/fused_quant_scatter_copy_sparse_flash_attention/op_kernel"
    )
    source = root / "tests/csrc/operators/attention/test_a5_copy_sfa_schedule.cpp"
    executable = tmp_path / "check_copy_sfa_schedule"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(include),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


@pytest.mark.parametrize(
    "device,graph", [("cpu", False), ("npu", False), ("npu", True)]
)
@pytest.mark.parametrize("width,prefix", [(1, 0), (4, 126)], ids=["q1", "q4"])
def test_a5_short_staging_matches_preloaded_attention(device, graph, width, prefix):
    """Compare a single token and a Q4 request accepting one row across a page."""
    if device == "npu":
        pytest.importorskip("torch_npu")
        import acl

        if not torch.npu.is_available() or acl.rt.get_device_info(0, 601) != (3510, 0):
            pytest.skip("Ascend950 required")
    impl = "ascendc" if device == "npu" else "torch_native"
    page = 128
    rng = torch.Generator().manual_seed(197)

    def packed_rows(rows):
        latent = torch.randn(rows, 512, generator=rng).to(torch.float8_e4m3fn)
        rope = torch.randn(rows, 64, generator=rng).to(torch.bfloat16)
        scales = torch.ones(rows, 4)
        return torch.cat([x.view(torch.int8) for x in (latent, rope, scales)], dim=1)

    def ids(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    logical = packed_rows(2 * page).to(device).view(2, page, 1, 656)
    host = logical.flip([0]).contiguous()
    hbm = torch.zeros_like(host)
    golden = logical.clone()
    query = torch.randn(width, 8, 576, generator=rng).to(device, torch.bfloat16)
    nope, rope = query[..., :512].contiguous(), query[..., 512:].contiguous()
    result = torch.empty_like(nope)
    expected = torch.empty_like(nope)
    topk = torch.full((width, 1, 2048), -1, dtype=torch.int32, device=device)
    ends, length = ids([width]), ids([prefix + width])
    hbm_pages, host_pages = ids([[0, 1]]), ids([[1, 0]])
    zero, row_misses = ids([0]), ids([0] * width)
    empty_union = torch.full((1, 32768), -1, dtype=torch.int32, device=device)
    current = torch.zeros((width, 656), dtype=torch.int8, device=device)
    position = torch.arange(prefix, prefix + width, device=device)
    extent = ids([-1, -1])
    req_to_token = torch.stack((ids([-1] * 256), ids([*range(128, 256), *range(128)])))
    resident = dict(
        current=current,
        positions=position,
        current_valid=torch.ones(width, device=device, dtype=torch.bool),
        request_slots=ids([1]),
        req_to_token=req_to_token,
        resident_end=extent,
        native_lim_plan=True,
    )

    def attention(cache, output, **kwargs):
        fused_quant_scatter_copy_sparse_flash_attention_v2(
            nope,
            rope,
            cache,
            host,
            host.data_ptr(),
            hbm_pages,
            host_pages,
            topk,
            topk,
            row_misses,
            ends,
            length,
            zero,
            empty_union,
            empty_union,
            zero,
            576**-0.5,
            output,
            impl=impl,
            **kwargs,
        )

    def update(first):
        length.fill_(first + width)
        position.copy_(torch.arange(first, first + width, device=device))
        current.copy_(packed_rows(width).to(device))
        for row in range(width):
            visible = first + row + 1
            topk[row, 0, :visible] = torch.arange(visible, device=device)

    update(prefix)
    if graph:
        attention(hbm, result, **resident)
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            attention(hbm, result, **resident)
        hbm.zero_()
        extent.fill_(-1)

    for first in (prefix, prefix + 1):
        update(first)
        stop = first + width
        golden.view(-1, 656)[first:stop].copy_(current)
        attention(golden, expected)
        if graph:
            capture.replay()
        else:
            attention(hbm, result, **resident)
        assert extent.tolist() == [-1, stop]
        assert torch.equal(hbm.view(-1, 656)[:stop], golden.view(-1, 656)[:stop])
        assert expected.isfinite().all() and expected.abs().max() > 0
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_torch_native_c8_uses_current_rows_and_group_scales():
    def packed(scale):
        latent = torch.ones((1, 512)).to(torch.float8_e4m3fn)
        rope = torch.zeros((1, 64), dtype=torch.bfloat16)
        return torch.cat([t.view(torch.int8) for t in (latent, rope, scale[None])], 1)

    def ids(values):
        return torch.tensor(values, dtype=torch.int32)

    host_scales = torch.tensor([1.0, 2.0, 0.5, 8.0])
    current_scales = torch.tensor([[3.0, 2.0, 1.5, 12.0], [5.0, 6.0, 2.5, 10.0]])
    host = torch.zeros((2, 128, 1, 656), dtype=torch.int8)
    host[1, 0] = packed(host_scales)
    hbm = torch.zeros_like(host)
    current = torch.cat([packed(scale) for scale in current_scales])
    query = torch.zeros((3, 8, 512), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    topk = torch.full((3, 1, 2048), -1, dtype=torch.int32)
    topk[0, 0, :2], topk[1, 0, :2] = ids([1, 127]), ids([1, 126])
    miss_sources = torch.full((2, 32768), -1, dtype=torch.int32)
    miss_destinations = torch.full_like(miss_sources, -1)
    miss_sources[0, :3] = ids([0, 128, 129])
    miss_destinations[0, :3] = ids([1, 127, 126])
    extents = ids([-1, 127])
    fused_quant_scatter_copy_sparse_flash_attention_v2(
        query,
        torch.zeros((3, 8, 64), dtype=query.dtype),
        hbm,
        host,
        0,
        ids([[1], [0]]),
        ids([[1, 0], [0, 1]]),
        topk,
        topk,
        ids([1, 1, 0]),
        ids([2, 3]),
        ids([129, 0]),
        ids([128, 0]),
        miss_sources,
        miss_destinations,
        ids([3, 0]),
        0.1,
        output,
        current=torch.cat((current, torch.zeros_like(current[:1]))),
        positions=torch.tensor([128, 129, 0]),
        current_valid=torch.tensor([True, True, False]),
        request_slots=ids([1, 0]),
        req_to_token=torch.full((2, 256), -1, dtype=torch.int32),
        resident_end=extents,
        native_lim_plan=True,
        impl="torch_native",
    )
    torch.testing.assert_close(hbm[1, 1, 0], host[1, 0, 0])
    torch.testing.assert_close(hbm[1, [127, 126], 0], current)
    expected = ((host_scales + current_scales) / 2).repeat_interleave(128, dim=1)
    torch.testing.assert_close(output[:2, 0], expected.to(output.dtype))
    assert not output[2].any()
    assert extents.tolist() == [-1, -1]
