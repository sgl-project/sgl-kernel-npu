import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_skip_acc_check
from tests.python.utils.performance import check_performance


def generate_data(mode, q, batch):
    lengths = ([2, 0, 3] if mode == "compact" else [2, 0, 1]) + [1] * (batch - 3)
    prefill = mode in ("compact", "padded")
    tokens = (
        sum(lengths) if mode == "compact" else batch * 3 if prefill else batch * q + 2
    )
    req = torch.tensor([1, 0, 5] + list(range(6, batch + 3)), dtype=torch.int32)
    seq = torch.tensor([10, 5, 8] + [9] * (batch - 3), dtype=torch.int32)
    positions = torch.arange(tokens, dtype=torch.int32) + 5
    loc = torch.arange(tokens, dtype=torch.int32) - 1
    if tokens > 3:
        loc[3] = 100
    ext = torch.tensor(lengths, dtype=torch.int32)
    prefix = torch.tensor([3, 4], dtype=torch.int32)
    return dict(
        tokens=tokens,
        device=req.device,
        request_ids=req,
        seq_lens=seq,
        positions=positions,
        out_cache_loc=loc,
        request_rows=20,
        context_len=20,
        host_capacity=50,
        query_width=q,
        extend_seq_lens=ext if prefill else None,
        extend_prefix_lens=prefix if prefill else None,
        compact_prefill=mode == "compact",
    ), lengths


def expected_rows(data, lengths, mode, q):
    req, seq, loc, positions = (
        data[k].cpu().tolist()
        for k in ("request_ids", "seq_lens", "out_cache_loc", "positions")
    )
    prefix = [3, 4]
    tokens, batch = data["tokens"], len(req)
    expected_pos, expected_valid = [], []
    for t in range(tokens):
        if mode in ("compact", "padded"):
            if mode == "compact":
                row = next(i for i in range(batch) if t < sum(lengths[: i + 1]))
                offset = t - sum(lengths[:row])
            else:
                row, offset = divmod(t, tokens // batch)
            pos = (prefix[row] if row < len(prefix) else 0) + offset
            live = req[row] > 0 and offset < lengths[row] and 0 <= pos < 20
        else:
            row = t // q
            pos = positions[t] if row < batch else -1
            live = row < batch and 0 < req[row] < 20 and seq[row] > 0 and 0 <= pos < 20
        expected_pos.append(pos)
        expected_valid.append(bool(live and 0 <= loc[t] < 50))
    return (
        torch.arange(tokens, device="cpu"),
        torch.tensor(loc, dtype=torch.int64, device="cpu"),
        torch.tensor(expected_pos, dtype=torch.int64, device="cpu"),
        torch.tensor(expected_valid, dtype=torch.bool, device="cpu"),
    )


@pytest.mark.parametrize(
    "mode,q,batch",
    [
        pytest.param("decode", 1, 16, id="decode-b16-performance"),
        pytest.param("verify", 4, 3, id="verify-q4-invalid-tail"),
        pytest.param("verify", 14, 3, id="verify-q14-block-tail"),
        pytest.param("compact", 1, 3, id="compact-zero-length-short-prefix"),
        pytest.param("padded", 1, 3, id="padded-zero-length-short-prefix"),
    ],
)
def test_prepare_sparse_kv_offload_rows(mode, q, batch):
    from sgl_kernel_npu.operators.mem_cache import prepare_sparse_kv_offload_rows

    data, lengths = generate_data(mode, q, batch)
    executor = check_accuracy(prepare_sparse_kv_offload_rows, kwargs=data)
    if get_skip_acc_check() is not True:
        expected = expected_rows(data, lengths, mode, q)
        assert_accuracy(tuple(t.cpu() for t in executor()), expected)
    check_performance(executor, enable_perf=mode == "decode" and batch == 16)
