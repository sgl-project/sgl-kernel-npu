"""Exact model-contract QSA Top-K and graph regression tests.

Migrated from the frozen qsa-model-v1 sandbox. Multi-GiB artificial address
and maximum-capacity stress cases remain in the sandbox, outside routine CI.
All device-value validation below is test-only, outside graph capture.
"""

import pytest
import torch
import torch_npu
from sgl_kernel_npu.qwen3_8_flash_next.qsa_topk import fast_topk, select_implementation

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
)


def check(score, lengths, starts, result, k):
    assert result.shape == (score.shape[0], k)
    assert result.dtype == torch.int32 and result.device == score.device
    assert result.is_contiguous()
    lens = lengths.cpu().long()
    begin = torch.zeros_like(lens) if starts is None else starts.cpu().long()
    output = result.cpu().long()
    for base in range(0, score.shape[0], 32):
        cpu = score[base : base + 32].cpu()
        ls, ss = lens[base : base + 32], begin[base : base + 32]
        ids = output[base : base + 32]
        # Batch the large homogeneous prefill oracle without changing tolerance.
        if ls.numel() and bool((ls == ls[0]).all() & (ss == ss[0]).all()) and ls[0] > k:
            l, s = int(ls[0]), int(ss[0])
            assert ((ids >= 0) & (ids < l)).all()
            ordered = ids.sort(1).values
            assert not (ordered[:, 1:] == ordered[:, :-1]).any()
            valid = cpu[:, s : s + l]
            torch.testing.assert_close(
                valid.gather(1, ids).sort(1).values,
                valid.topk(k, dim=1).values.sort(1).values,
                rtol=0,
                atol=0,
            )
            continue
        for i in range(len(ls)):
            l, s = int(ls[i]), int(ss[i])
            n = min(l, k)
            chosen = ids[i, :n]
            assert ((chosen >= 0) & (chosen < l)).all()
            assert chosen.unique().numel() == n
            assert (ids[i, n:] == -1).all()
            if l <= k:
                assert torch.equal(chosen, torch.arange(l))
            else:
                valid = cpu[i, s : s + l]
                torch.testing.assert_close(
                    valid[chosen].sort().values,
                    valid.topk(k).values.sort().values,
                    rtol=0,
                    atol=0,
                )


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "pattern", ["random", "constant", "binary", "close", "signed_zero", "subnormal"]
)
@pytest.mark.parametrize("width", [8193, 262145])
def test_model_values(k, pattern, width):
    torch.manual_seed(11)
    cpu = torch.randn(8, width)
    if pattern == "constant":
        cpu.zero_()
    elif pattern == "binary":
        cpu = (cpu > 0).float()
    elif pattern == "close":
        cpu = torch.arange(width).float().expand(8, -1).clone() * 2**-23 + 1
    elif pattern == "signed_zero":
        cpu.zero_()
        cpu[:, ::2] = -0.0
    elif pattern == "subnormal":
        cpu = torch.arange(width).float().expand(8, -1).clone() * 2**-149
    lengths = torch.tensor(
        [0, 1, k - 1, k, k + 1, width - 17, k + 3, width - 17],
        dtype=torch.int32,
        device="npu",
    )
    starts = torch.arange(8, dtype=torch.int32, device="npu")
    for row, (s, l) in enumerate(zip(starts.cpu().tolist(), lengths.cpu().tolist())):
        cpu[row, :s] = float("nan")
        cpu[row, s + l :] = float("inf")
    backing = torch.empty(8, width + 19, device="npu")
    x = backing[:, :width]
    x.copy_(cpu)
    ls, ss = lengths.clone(), starts.clone()
    check(x, lengths, starts, fast_topk(x, lengths, k, starts), k)
    assert torch.equal(cpu.view(torch.int32), x.cpu().contiguous().view(torch.int32))
    assert torch.equal(ls, lengths) and torch.equal(ss, starts)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "rows,width",
    [
        (0, 0),
        (3, 0),
        (4, 17),
        (4097, 513),
        (8, 4096),
        (8, 65536),
        (2, 262144),
        (2, 262145),
        (2, 1048577),
    ],
)
def test_model_sizes(k, rows, width):
    torch.manual_seed(17)
    x = torch.randn(rows, width, device="npu")
    lens = torch.full((rows,), width, dtype=torch.int32, device="npu")
    check(x, lens, None, fast_topk(x, lens, k), k)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("rows", [4095, 4096, 65534, 65535, 65536, 65537])
def test_model_grid(k, rows):
    x = torch.empty(rows, 17, device="npu")
    lens = (torch.arange(rows, device="npu") % 18).int()
    out = fast_topk(x, lens, k).cpu()
    cols = torch.arange(k)[None, :]
    assert torch.equal(out, torch.where(cols < lens.cpu()[:, None], cols, -1).int())


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("rows,width", [(129, 16384), (513, 65537)])
def test_model_large_prefill(k, rows, width):
    torch.manual_seed(97)
    x = torch.randn(rows, width, device="npu")
    lens = torch.full((rows,), width, dtype=torch.int32, device="npu")
    check(x, lens, None, fast_topk(x, lens, k), k)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("shape", [(4, 8193), (129, 4096), (257, 131073), (3, 262145)])
def test_model_replay_individual_inputs(k, shape):
    rows, width = shape
    backing = torch.randn(rows, width + 19, device="npu")
    x = backing[:, :width]
    lens = torch.full((rows,), width - 8, dtype=torch.int32, device="npu")
    starts = torch.zeros_like(lens)
    for _ in range(2):
        check(x, lens, starts, fast_topk(x, lens, k, starts), k)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = fast_topk(x, lens, k, starts)
    # Change exactly one logical input at each step; no host-value dispatch.
    for step in range(6):
        if step == 0:
            x.zero_()
        elif step == 1:
            starts.fill_(7)
        elif step == 2:
            lens.fill_(k + 3)
            lens[::3] = 0
            lens[1::3] = k
        elif step == 3:
            x.normal_()
        elif step == 4:
            starts.zero_()
        else:
            lens.fill_(width - 8)
        g.replay()
        check(x, lens, starts, out, k)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "rows,width",
    [
        (2047, 16384),
        (2048, 16384),
        (2049, 16384),
        (2, 262143),
        (2, 262144),
        (2, 262145),
    ],
)
def test_model_replay_boundaries(k, rows, width):
    x = torch.randn(rows, width, device="npu")
    lens = torch.full((rows,), width - 4, dtype=torch.int32, device="npu")
    starts = torch.zeros_like(lens)
    check(x, lens, starts, fast_topk(x, lens, k, starts), k)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = fast_topk(x, lens, k, starts)
    for step in range(3):
        starts.fill_(step)
        lens.fill_(width - 4 if step != 1 else k + 3)
        lens[::3] = 0
        lens[1::3] = k
        if step == 1:
            x.zero_()
        else:
            x.normal_()
        g.replay()
        check(x, lens, starts, out, k)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("rows", [1, 39, 40, 41, 127, 128, 129])
@pytest.mark.parametrize("width", [17, 4096])
def test_model_replay_without_starts(k, rows, width):
    x = torch.randn(rows, width, device="npu")
    lens = torch.full((rows,), width, dtype=torch.int32, device="npu")
    for _ in range(2):
        fast_topk(x, lens, k)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = fast_topk(x, lens, k)
    for count in (0, width, min(k, width)):
        x.normal_()
        lens.fill_(count)
        g.replay()
        check(x, lens, None, out, k)


@pytest.mark.parametrize("field", ["lengths", "starts"])
def test_model_reject_strided_bounds(field):
    x = torch.randn(4, 8193, device="npu")
    lens = torch.full((4,), 4096, dtype=torch.int32, device="npu")
    starts = torch.zeros_like(lens)
    backing = torch.zeros(8, dtype=torch.int32, device="npu")
    if field == "lengths":
        backing[::2] = lens
        lens = backing[::2]
    else:
        starts = backing[::2]
    with pytest.raises(ValueError, match="contiguous int32"):
        fast_topk(x, lens, 512, starts)


@pytest.mark.parametrize("rows,width", [(0, 2**24 + 1), (0, 2**31 + 4096)])
def test_model_reject_width(rows, width):
    # Zero-stride row with zero rows avoids allocating obsolete huge storage.
    x = torch.empty(rows, width, device="npu")
    lens = torch.zeros(rows, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="maximum"):
        fast_topk(x, lens, 512)


@pytest.mark.parametrize(
    "kind",
    [
        "k",
        "dtype",
        "score_rank",
        "column_stride",
        "bounds_dtype",
        "bounds_shape",
        "bounds_device",
    ],
)
def test_model_reject_metadata(kind):
    x = torch.empty(4, 4096, device="npu")
    lens = torch.zeros(4, dtype=torch.int32, device="npu")
    k = 512
    if kind == "k":
        k = 1024
    elif kind == "dtype":
        x = x.half()
    elif kind == "score_rank":
        x = x[0]
    elif kind == "column_stride":
        x = x[:, ::2]
    elif kind == "bounds_dtype":
        lens = lens.long()
    elif kind == "bounds_shape":
        lens = lens[:3]
    elif kind == "bounds_device":
        lens = lens.cpu()
    with pytest.raises(ValueError):
        fast_topk(x, lens, k)


@pytest.mark.parametrize(
    "rows,columns,k,path",
    [
        (0, 4096, 512, "shortcut"),
        (8, 256, 512, "shortcut"),
        (8, 4096, 512, "tiled"),
        (2047, 16384, 512, "tiled"),
        (2048, 16384, 512, "hybrid"),
        (2, 262144, 2048, "tiled"),
        (2, 262145, 2048, "hybrid"),
    ],
)
def test_shape_dispatch(rows, columns, k, path):
    assert select_implementation(rows, columns, k) == path
