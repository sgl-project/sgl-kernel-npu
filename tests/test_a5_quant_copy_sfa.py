import pytest
import torch


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("width,prefix", [(1, 0), (4, 126)], ids=["q1", "q4"])
def test_a5_short_staging_matches_preloaded_attention(graph, width, prefix):
    """Compare a single token and a Q4 request accepting one row across a page."""
    pytest.importorskip("torch_npu")
    import acl
    import sgl_kernel_npu  # noqa: F401

    if not torch.npu.is_available() or acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("Ascend950 required")
    device = "npu:0"
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
        torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2(
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
