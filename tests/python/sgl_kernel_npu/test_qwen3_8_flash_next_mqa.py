"""Model-contract MQA regression tests.

Ported from the frozen qsa_mqa v1.1 sandbox. Legacy generic dtype/stride,
custom-scale and independent-width coverage is intentionally replaced by
metadata rejection tests. The old tests remain available in Git history.
CPU FP64 is an independent oracle; device Torch FP32 is a separate reference.
"""
import math
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.qwen3_8_flash_next import mqa as wrapper

@dataclass(frozen=True)
class Case:
    kind: str
    rows: int
    width: int
    pattern: str = "mixed"
    stress: bool = False


CASES = {
    "packed_empty": Case("packed", 0, 17),
    "packed_no_keys": Case("packed", 3, 0),
    "packed_q1_prefix": Case("packed", 1, 1024),
    "packed_q1_long_prefix": Case("packed", 1, 65536, "mixed", True),
    "packed_tail": Case("packed", 33, 257),
    "packed_multi_request": Case("packed", 64, 48, "two_requests"),
    "packed_chunk": Case("packed", 4096, 1024, "fresh_chunk", True),
    "packed_long_prefix": Case("packed", 512, 65536, "mixed", True),
    "packed_multi_long": Case("packed", 256, 131072, "two_requests", True),
    "paged_empty": Case("paged", 0, 16),
    "paged_zero_width": Case("paged", 3, 0),
    "paged_decode_b1": Case("paged", 1, 1024),
    "paged_decode_b8": Case("paged", 8, 1024),
    "paged_verify_b8_w4": Case("paged", 32, 1040, "verify"),
    "paged_old_limit_129": Case("paged", 129, 128),
    "paged_graph_short": Case("paged", 8, 65536, "short", True),
    "paged_long": Case("paged", 32, 65536, "mixed", True),
    "packed_r1_w257": Case("packed", 1, 257),
    "packed_r7_w511": Case("packed", 7, 511),
    "packed_r16_w512": Case("packed", 16, 512),
    "packed_r33_w513": Case("packed", 33, 513),
    "packed_r64_w4096": Case("packed", 64, 4096),
    "packed_r128_w16384": Case("packed", 128, 16384, "two_requests", True),
    "packed_r512_w256": Case("packed", 512, 256),
    "packed_r1024_w4096": Case("packed", 1024, 4096, "mixed", True),
    "packed_r7_w1023": Case("packed", 7, 1023),
    "packed_r16_w1024": Case("packed", 16, 1024),
    "packed_r33_w1025": Case("packed", 33, 1025),
}


def make_case(name, device="cpu", seed=73):
    c = CASES[name]
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn(c.rows, 4, 128, generator=rng, dtype=torch.bfloat16)
    if c.kind == "packed":
        keys = torch.randn(c.width, 1, 128, generator=rng, dtype=torch.bfloat16)
        starts = torch.zeros(c.rows, dtype=torch.int32)
        ends = torch.full((c.rows,), c.width, dtype=torch.int32)
        if c.pattern == "two_requests":
            starts[c.rows // 2:] = c.width // 2
            ends[:c.rows // 2] = c.width // 2
        elif c.pattern == "fresh_chunk":
            ends = ((torch.arange(c.rows) + 1) // 4).to(torch.int32)
        elif c.rows > 1:
            ends = torch.linspace(0, c.width, c.rows).to(torch.int32)
        args = (q, keys, starts, ends)
    else:
        pages = c.width // 16
        requests = c.rows // 4 if c.pattern == "verify" else c.rows
        # Unique pages within a request; verify rows share the request mapping.
        table = torch.arange(1, requests * pages + 1, dtype=torch.int32).reshape(requests, pages)
        if c.pattern == "verify":
            table = table.repeat_interleave(4, dim=0)
            lengths = ((4089 + torch.arange(4) + 1) // 4).repeat(requests).to(torch.int32)
        else:
            limit = min(c.width, 33) if c.pattern == "short" else c.width
            lengths = torch.linspace(0, limit, c.rows).to(torch.int32)
            if c.rows == 1:
                lengths.fill_(limit)
        cache = torch.randn(max(1, requests * pages + 1), 16, 1, 128,
                            generator=rng, dtype=torch.bfloat16)
        args = (q, cache, table, lengths, c.width)
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in args)


def expected(kind, args):
    q, keys, first, second, *extra = [
        x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args
    ]
    width = keys.shape[0] if kind == "packed" else extra[0]
    out = torch.full((q.shape[0], width), -float("inf"), dtype=torch.float64)
    for row in range(q.shape[0]):
        if kind == "packed":
            start, end = int(first[row]), int(second[row])
            selected = keys[start:end, 0].double()
        else:
            start, end = 0, int(second[row])
            positions = torch.arange(end)
            pages = first[row, positions // keys.shape[1]].long()
            selected = keys[pages, positions % keys.shape[1], 0].double()
        values = torch.zeros(end - start, dtype=torch.float64)
        for head in range(q.shape[1]):
            values += (selected @ q[row, head].double()).clamp_min(0)
        out[row, start:end] = values / math.sqrt(q.shape[-1])
    return out.float()


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits



@pytest.mark.parametrize("name", list(CASES))
def test_model(name):
    kind = CASES[name].kind
    args = make_case(name, device="npu")
    saved = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    out = getattr(wrapper, kind)(*args)
    check(kind, args, out)
    assert out.is_contiguous()
    for x, before in zip(args, saved):
        torch.testing.assert_close(x, before, atol=0, rtol=0)

def check(kind, args, out, reference=True):
    oracle = expected(kind, args)
    assert out.shape == oracle.shape and out.dtype == torch.float32
    actual = out.cpu()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(oracle))
    torch.testing.assert_close(actual, oracle, atol=2e-5, rtol=2e-5)
    if reference:
        fn = torch_qsa_mqa_prefill if kind == "packed" else torch_qsa_mqa_decode
        torch.testing.assert_close(out, fn(*args), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("rows,width", [(1,1), (3,31), (4,32), (5,33),
                                       (7,63), (8,64), (9,65),
                                       (15,127), (16,128), (17,129), (33,257)])
@pytest.mark.parametrize("kind", ["packed", "paged"])
def test_tile_boundaries(kind, rows, width):
    rng = torch.Generator().manual_seed(109)
    q = torch.randn(rows, 4, 128, generator=rng).bfloat16()
    if kind == "packed":
        k = torch.randn(width, 1, 128, generator=rng).bfloat16()
        starts = torch.full((rows,), width // 3, dtype=torch.int32)
        ends = torch.linspace(width // 3, width, rows).to(torch.int32)
        ends[-1] = width
        args = (q, k, starts, ends)
    else:
        pages = (width + 15) // 16
        width = pages * 16
        k = torch.randn(pages + 2, 16, 1, 128, generator=rng).bfloat16()
        # Reversed maps share physical pages across rows without identity addressing.
        table = torch.arange(pages, 0, -1, dtype=torch.int32).repeat(rows, 1)
        choices = torch.tensor([0,1,15,16,17,31,32,33,63,64,65,
                                127,128,129,255,256,257], dtype=torch.int32)
        lengths = choices[torch.arange(rows) % choices.numel()].clamp_max(width)
        lengths[-1] = width
        args = (q, k, table, lengths, width)
    args = tuple(x.to("npu") if isinstance(x, torch.Tensor) else x for x in args)
    saved = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    check(kind, args, getattr(wrapper, kind)(*args))
    for value, before in zip(args, saved):
        assert torch.equal(value, before)


def test_unused_pages_and_valid_zero():
    q = torch.zeros(6, 4, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.ones(4, 16, 1, 128, dtype=torch.bfloat16, device="npu")
    cache[0] = float("nan")
    lengths = torch.tensor([0,1,15,16,17,33], dtype=torch.int32, device="npu")
    table = torch.tensor([[1,2,3,0]] * 6, dtype=torch.int32, device="npu")
    positions = torch.arange(4, device="npu")[None, :] * 16
    table.masked_fill_(positions >= lengths[:, None], 123456)
    args = (q, cache, table, lengths, 64)
    check("paged", args, wrapper.paged(*args), reference=False)


@pytest.mark.parametrize("seed", [73, 997])
@pytest.mark.parametrize("kind", ["packed", "paged"])
def test_normalized_and_cancelling_scores(kind, seed):
    name = "packed_tail" if kind == "packed" else "paged_verify_b8_w4"
    args = list(make_case(name, seed=seed))
    for i in (0, 1):
        x = args[i].float()
        args[i] = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    # Opposite head pairs put positive and negative dots near the same ReLU boundary.
    args[0][:, 1] = -args[0][:, 0]
    args[0][:, 3] = -args[0][:, 2]
    args = tuple(x.to("npu") if isinstance(x, torch.Tensor) else x for x in args)
    check(kind, args, getattr(wrapper, kind)(*args))


@pytest.mark.parametrize("name", ["packed_multi_request", "packed_tail", "packed_q1_prefix",
                                  "packed_q1_long_prefix",
                                  "packed_r7_w1023", "packed_r16_w1024",
                                  "packed_r33_w1025",
                                  "paged_verify_b8_w4", "paged_old_limit_129",
                                  "paged_graph_short"])
def test_independent_graph_updates(name):
    kind = name.split("_")[0]
    args = make_case(name, device="npu")
    fn = getattr(wrapper, kind)
    originals = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
    check(kind, args, fn(*args))
    for _ in range(2):
        fn(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fn(*args)
    graph.replay()
    torch.npu.synchronize()
    check(kind, args, out)
    for change in ("q", "k", "first", "last", "empty", "restore"):
        for value, original in zip(args, originals):
            value.copy_(original)
        if change == "q":
            args[0].neg_()
        elif change == "k":
            args[1].neg_()
        elif change == "first":
            if kind == "packed":
                args[2].copy_(args[3])
            else:
                args[2].copy_(args[2].flip(0).roll(1, dims=1))
        elif change in ("last", "empty"):
            if kind == "packed":
                args[3].copy_(args[2])
            elif change == "empty":
                args[3].zero_()
            else:
                args[3].fill_(args[4])
        graph.replay()
        torch.npu.synchronize()
        check(kind, args, out)
        assert addresses == [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]


def test_out_of_scope_rejected():
    q, k, starts, ends = make_case("packed_tail", device="npu")
    for args in ((q.float(), k, starts, ends), (q[:, :3], k, starts, ends),
                 (q, k, starts.long(), ends), (q, k, starts, ends[:-1])):
        with pytest.raises(ValueError):
            wrapper.packed(*args)
    q, k, table, lengths, width = make_case("paged_decode_b8", device="npu")
    with pytest.raises(ValueError):
        wrapper.paged(q, k, table, lengths, width - 1)
