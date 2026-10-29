"""DeepSeek-V4 metadata accuracy, schedule properties and input restrictions."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Single-request decode is the independent reference's supported domain.
# Head counts and D=512 follow the DSV4 shared-KV interface; short KV lengths
# keep these synthetic fixtures small. They are not captured model batches.
perf_shape = [("BSND", 64, False, 1, 128)]


def generate_data(heads=64, query_lengths=(1,), kv_lengths=(128,), **overrides):
    cu_q = torch.tensor([0, *query_lengths], dtype=torch.int32, device="cpu").cumsum(
        0, dtype=torch.int32
    )
    kwargs = dict(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=512,
        layout_q="TND",
        layout_kv="PA_ND",
        cu_seqlens_q=cu_q,
        seqused_kv=torch.tensor(kv_lengths, dtype=torch.int32, device="cpu"),
        batch_size=len(query_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_kv=max(kv_lengths),
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    return kwargs | overrides


def assert_valid_schedule(metadata, batch_size):
    assert metadata.device == torch.device("npu", torch.npu.current_device())
    assert metadata.dtype == torch.int32
    assert metadata.shape == (1024,)
    host = metadata.cpu()
    cube = host[: 36 * 9].view(36, 9)
    vector = host[36 * 9 : 36 * 9 + 72 * 8].view(72, 8)
    enabled = int((cube[:, 0] == 1).sum())
    assert enabled >= 1
    assert torch.all(cube[:enabled, 0] == 1)
    assert torch.all(cube[enabled:, 0] == 0)
    assert cube[enabled - 1, 4].item() == batch_size
    assert torch.all(vector[:, 0] == 0)  # Flash Decode splitting is disabled.


@pytest.mark.parametrize(
    ("layout", "heads", "has_cmp", "cmp_ratio", "kv_len"),
    [
        pytest.param("BSND", 64, False, 1, 128, id="dsv4-decode-64-heads"),
        pytest.param("TND", 64, True, 4, 128, id="dsv4-compressed-ratio4"),
        pytest.param("TND", 128, False, 1, 256, id="dsv4-decode-128-heads"),
    ],
)
def test_host_metadata_accuracy(layout, heads, has_cmp, cmp_ratio, kv_len):
    kwargs = generate_data(
        heads=heads,
        kv_lengths=(kv_len,),
        layout_q=layout,
        cmp_ratio=cmp_ratio,
        has_cmp_kv=has_cmp,
    )
    executor = check_accuracy(
        kv_quant_sparse_attn_sharedkv_metadata_host, kwargs=kwargs
    )
    check_performance(
        executor,
        enable_perf=(layout, heads, has_cmp, cmp_ratio, kv_len) in perf_shape,
    )


@pytest.mark.parametrize(
    ("query_lengths", "kv_lengths", "cmp_ratio", "has_cmp", "cmp_topk"),
    [
        pytest.param(
            (1,) * 8,
            (128, 256, 512, 1024, 2048, 4096, 8192, 12345),
            1,
            False,
            0,
            id="mixed-request-lengths",
        ),
        pytest.param((1,) * 8, (4096,) * 8, 4, True, 512, id="compressed-ratio4"),
        pytest.param((1,) * 8, (4096,) * 8, 128, True, 0, id="compressed-ratio128"),
        pytest.param((32768,), (32768,), 1, False, 0, id="long-prefill-multiple-cores"),
        pytest.param((1,), (65536,), 1, False, 0, id="long-context-decode"),
    ],
)
def test_schedule_properties(query_lengths, kv_lengths, cmp_ratio, has_cmp, cmp_topk):
    # General schedules have no independent Python load-balancer reference.
    # Check their structure and determinism instead of comparing host aliases.
    kwargs = generate_data(
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        cmp_ratio=cmp_ratio,
        has_cmp_kv=has_cmp,
        cmp_topk=cmp_topk,
    )
    actual = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs)
    assert_valid_schedule(actual, len(query_lengths))
    repeated = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    if max(query_lengths) > 1:
        cube = actual.cpu()[: 36 * 9].view(36, 9)
        assert int((cube[:, 0] == 1).sum()) > 1


def test_infers_batch_size():
    kwargs = generate_data()
    explicit = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs)
    inferred = kv_quant_sparse_attn_sharedkv_metadata_host(
        **(kwargs | {"batch_size": 0})
    )
    torch.testing.assert_close(inferred, explicit, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        pytest.param(
            "kv_quant_mode", 0, "expected kv_quant_mode", id="unsupported-quant-mode"
        ),
        pytest.param(
            "tile_size", 128, "expected kv_quant_mode", id="unsupported-tile-size"
        ),
        pytest.param(
            "rope_head_dim", 128, "expected kv_quant_mode", id="unsupported-rope-dim"
        ),
        pytest.param("cmp_topk", 64, "A5 scheduling failed", id="unsupported-topk"),
        pytest.param("cmp_ratio", 7, "A5 scheduling failed", id="unsupported-ratio"),
    ],
)
def test_rejects_unsupported_configuration(name, value, message):
    kwargs = generate_data(has_cmp_kv=True, cmp_ratio=4, cmp_topk=512)
    with pytest.raises(RuntimeError, match=message):
        kv_quant_sparse_attn_sharedkv_metadata_host(**(kwargs | {name: value}))


@pytest.mark.parametrize("location", ["cpu-int64", "npu-int32"])
def test_requires_cpu_int32_lengths(location):
    kwargs = generate_data()
    cu_q = kwargs["cu_seqlens_q"]
    kwargs["cu_seqlens_q"] = (
        cu_q.to(dtype=torch.int64) if location == "cpu-int64" else cu_q.to("npu")
    )
    with pytest.raises(RuntimeError, match="contiguous CPU int32"):
        kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs)
