import math

import pytest
import sgl_kernel_npu  # noqa: F401  Registers torch.ops.npu.sgl_sparse_flash_attention.
import torch
import torch_npu  # noqa: F401  Registers torch.npu.
from utils import require_npu_op

pytestmark = [
    pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required"),
    require_npu_op("sgl_sparse_flash_attention"),
]


def _dsa_inputs():
    """A single-request DSA decode case with every KV token selected."""
    torch.manual_seed(2026)
    device = torch.device("npu:0")
    tokens, q_heads, nope_dim, rope_dim = 1, 8, 512, 64
    block_count, block_size = 8, 16
    seq_len = block_count * block_size
    query = torch.randn(tokens, q_heads, nope_dim, dtype=torch.bfloat16, device=device)
    query_rope = torch.randn(
        tokens, q_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        block_count, block_size, 1, nope_dim, dtype=torch.bfloat16, device=device
    )
    key_rope = torch.randn(
        block_count, block_size, 1, rope_dim, dtype=torch.bfloat16, device=device
    )
    sparse_indices = torch.full((tokens, 1, 2048), -1, dtype=torch.int32, device=device)
    sparse_indices[..., :seq_len] = torch.arange(
        seq_len, dtype=torch.int32, device=device
    )
    return {
        "query": query,
        "key": key,
        "value": key,
        "sparse_indices": sparse_indices,
        "scale_value": 1.0 / math.sqrt(nope_dim + rope_dim),
        "block_table": torch.tensor(
            [[3, 0, 6, 1, 7, 4, 2, 5]], dtype=torch.int32, device=device
        ),
        "actual_seq_lengths_query": torch.ones(1, dtype=torch.int32, device=device),
        "actual_seq_lengths_kv": torch.full(
            (1,), seq_len, dtype=torch.int32, device=device
        ),
        "query_rope": query_rope,
        "key_rope": key_rope,
    }


def _paged_to_tnd(tensor, block_table):
    return tensor.index_select(0, block_table[0].long()).flatten(0, 1)


def _call(inputs, *, return_softmax_lse=False):
    return torch.ops.npu.sgl_sparse_flash_attention(
        **inputs,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=return_softmax_lse,
    )


def _is_a5():
    return torch.npu.is_available() and "950" in torch.npu.get_device_name(0)


def _dense_reference(inputs):
    query = torch.cat((inputs["query"], inputs["query_rope"]), dim=-1).float()
    key_nope = inputs["key"]
    key_rope = inputs["key_rope"]
    value = inputs["value"]
    if inputs["block_table"] is not None:
        key_nope = _paged_to_tnd(key_nope, inputs["block_table"])
        key_rope = _paged_to_tnd(key_rope, inputs["block_table"])
        value = _paged_to_tnd(value, inputs["block_table"])
    key = torch.cat((key_nope, key_rope), dim=-1).reshape(-1, 1, 576).float()
    value = value.reshape(-1, 1, 512).float()
    scores = torch.einsum("qhd,shd->qhs", query, key) * inputs["scale_value"]
    return torch.einsum("qhs,shd->qhd", torch.softmax(scores, dim=-1), value).to(
        torch.bfloat16
    )


def test_dsa_sparse_flash_attention_matches_dense_reference():
    inputs = _dsa_inputs()
    output, softmax_max, softmax_sum = _call(inputs)
    torch.npu.synchronize()

    torch.testing.assert_close(output, _dense_reference(inputs), rtol=1e-2, atol=1e-2)
    assert softmax_max.numel() == 0
    assert softmax_sum.numel() == 0


def test_dsa_softmax_lse_matches_dense_reference_with_tnd_kv():
    inputs = _dsa_inputs()
    inputs["key"] = _paged_to_tnd(inputs["key"], inputs["block_table"])
    inputs["value"] = _paged_to_tnd(inputs["value"], inputs["block_table"])
    inputs["key_rope"] = _paged_to_tnd(inputs["key_rope"], inputs["block_table"])
    inputs["block_table"] = None

    output, softmax_max, softmax_sum = torch.ops.npu.sgl_sparse_flash_attention(
        **inputs,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()

    query = torch.cat((inputs["query"], inputs["query_rope"]), dim=-1).float()
    key = torch.cat((inputs["key"], inputs["key_rope"]), dim=-1).float()
    scores = torch.einsum("qhd,shd->qhs", query, key) * inputs["scale_value"]
    expected_lse = torch.logsumexp(scores, dim=-1)
    actual_lse = (softmax_max + torch.log(softmax_sum)).squeeze(0)

    torch.testing.assert_close(output, _dense_reference(inputs), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-3, atol=1e-3)


def test_dsa_dcp_lse_matches_dense_reference():
    inputs = _dsa_inputs()
    output, softmax_max, softmax_sum = _call(inputs, return_softmax_lse=True)
    torch.npu.synchronize()

    query = torch.cat((inputs["query"], inputs["query_rope"]), dim=-1).float()
    key_nope = _paged_to_tnd(inputs["key"], inputs["block_table"])
    key_rope = _paged_to_tnd(inputs["key_rope"], inputs["block_table"])
    key = torch.cat((key_nope, key_rope), dim=-1).reshape(-1, 1, 576).float()
    scores = torch.einsum("qhd,shd->qhs", query, key) * inputs["scale_value"]
    expected_lse = torch.logsumexp(scores, dim=-1)
    actual_lse = (softmax_max + torch.log(softmax_sum)).squeeze(0)

    torch.testing.assert_close(output, _dense_reference(inputs), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_a5(), reason="Regression for the A5 SFA kernel")
@pytest.mark.parametrize("selected_count", [0, 1, 127, 128, 129, 257])
def test_dsa_dcp_padded_indices_lse_on_a5(selected_count):
    torch.manual_seed(928)
    device = torch.device("npu:0")
    heads, kv_length, capacity = 64, 512, 2048
    query = torch.randn(1, heads, 512, dtype=torch.bfloat16, device=device)
    query_rope = torch.randn(1, heads, 64, dtype=torch.bfloat16, device=device)
    key = torch.randn(kv_length, 512, dtype=torch.bfloat16)
    key_rope = torch.randn(kv_length, 64, dtype=torch.bfloat16)
    selected = torch.randperm(kv_length)[:selected_count].sort().values
    indices = torch.full((1, 1, capacity), -1, dtype=torch.int32)
    indices[0, 0, :selected_count] = selected.int()

    output, maximum, total = torch.ops.npu.sgl_sparse_flash_attention(
        query=query,
        key=key.reshape(4, 128, 1, 512).to(device),
        value=key.reshape(4, 128, 1, 512).to(device),
        sparse_indices=indices.to(device),
        scale_value=1.0 / 24,
        sparse_block_size=1,
        block_table=torch.arange(4, dtype=torch.int32).reshape(1, -1).to(device),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=device),
        actual_seq_lengths_kv=torch.tensor(
            [kv_length], dtype=torch.int32, device=device
        ),
        query_rope=query_rope,
        key_rope=key_rope.reshape(4, 128, 1, 64).to(device),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()

    output = output.cpu().float().reshape(heads, 512)
    lse = (maximum.cpu().float() + total.cpu().float().log()).reshape(heads)
    if selected_count == 0:
        assert torch.count_nonzero(output) == 0
        assert torch.isneginf(lse).all()
        return

    logits = (
        query.cpu()[0].float() @ key[selected].float().T
        + query_rope.cpu()[0].float() @ key_rope[selected].float().T
    ) / 24
    expected = logits.softmax(-1) @ key[selected].float()
    torch.testing.assert_close(output, expected, atol=0.03, rtol=0.01)
    torch.testing.assert_close(lse, logits.logsumexp(-1), atol=0.005, rtol=0.001)


@pytest.mark.skipif(not _is_a5(), reason="Packed MLA KV is supported by the A5 kernel")
@require_npu_op("sgl_kv_quant_sparse_flash_attention")
@pytest.mark.parametrize(
    ("heads", "page_ids", "long_count", "expand_query_rows"),
    [
        pytest.param(32, [2, 0, 1], 129, False, id="direct-l1"),
        pytest.param(128, [2, 0, 1], 129, False, id="split-g"),
        pytest.param(32, [2, 0, 4, 1, 3], 385, False, id="direct-l1-buffer-reuse"),
        pytest.param(32, [2, 0, 1], 129, True, id="verify-metadata"),
    ],
)
def test_dsa_dcp_packed_kv_shared_page_table(
    heads, page_ids, long_count, expand_query_rows
):
    torch.manual_seed(929)
    device = torch.device("npu:0")
    page_count, page_size = len(page_ids), 128
    query = (torch.randn(3, heads, 512) * 0.5).bfloat16()
    query_rope = (torch.randn(3, heads, 64) * 0.5).bfloat16()
    latent = torch.randn(page_count, page_size, 1, 4, 128).to(torch.float8_e4m3fn)
    scales = torch.rand(page_count, page_size, 1, 4) * 0.4 + 0.1
    rope = (torch.randn(page_count, page_size, 1, 64) * 0.4).bfloat16()
    packed = torch.cat(
        (
            latent.flatten(-2).view(torch.uint8),
            rope.view(torch.uint8),
            scales.view(torch.uint8),
        ),
        dim=-1,
    )

    # Three query rows with different valid prefixes share one request's page table.
    kv_length = (page_count - 1) * page_size + 4
    selected = ([3, 96, 127], list(range(page_size, page_size + long_count)), [])
    indices = torch.full((3, 1, 2048), -1, dtype=torch.int32)
    for row, ids in enumerate(selected):
        indices[row, 0, : len(ids)] = torch.tensor(ids, dtype=torch.int32)

    query_lengths = [1, 2, 3] if expand_query_rows else [3]
    block_tables = [page_ids] * len(query_lengths)
    scale_value = 1.0 / 24
    output, maximum, total = torch.ops.npu.sgl_kv_quant_sparse_flash_attention(
        query=query.to(device),
        query_rope=query_rope.to(device),
        packed_kv=packed.to(device),
        sparse_indices=indices.to(device),
        block_table=torch.tensor(block_tables, dtype=torch.int32, device=device),
        actual_seq_lengths_query=torch.tensor(
            query_lengths, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_kv=torch.full(
            (len(query_lengths),), kv_length, dtype=torch.int32, device=device
        ),
        scale_value=scale_value,
    )
    output = output.cpu().float()
    maximum = maximum.cpu().squeeze(0)
    total = total.cpu().squeeze(0)

    key = (latent.float() * scales.unsqueeze(-1)).bfloat16().float()
    key = key[page_ids].reshape(-1, 512)
    key_rope = rope[page_ids].reshape(-1, 64).float()
    for row, ids in enumerate(selected[:2]):
        scores = (
            query[row].float() @ key[ids].T + query_rope[row].float() @ key_rope[ids].T
        ) * scale_value
        expected = scores.softmax(dim=-1) @ key[ids]
        torch.testing.assert_close(output[row], expected, atol=0.02, rtol=0.03)
        torch.testing.assert_close(
            maximum[row], scores.amax(dim=-1), atol=0.005, rtol=0.01
        )
        torch.testing.assert_close(
            maximum[row] + total[row].log(),
            scores.logsumexp(dim=-1),
            atol=0.005,
            rtol=0.001,
        )
    assert torch.count_nonzero(output[2]) == 0
    assert torch.isneginf(maximum[2] + total[2].log()).all()


def test_dsa_dcp_lse_npu_graph_replay_matches_reference():
    inputs = _dsa_inputs()
    eager, eager_max, eager_sum = _call(inputs, return_softmax_lse=True)
    torch.npu.synchronize()

    staging = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    graph_output = torch.empty_like(eager)
    graph_max = torch.empty_like(eager_max)
    graph_sum = torch.empty_like(eager_sum)
    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    torch.npu.synchronize()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
        output, softmax_max, softmax_sum = _call(staging, return_softmax_lse=True)
        graph_output.copy_(output)
        graph_max.copy_(softmax_max)
        graph_sum.copy_(softmax_sum)
    torch.npu.synchronize()

    staging["query"].copy_(inputs["query"] * 0.5)
    staging["query_rope"].copy_(inputs["query_rope"] * 0.5)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        graph_output, _dense_reference(staging), rtol=1e-2, atol=1e-2
    )

    query = torch.cat((staging["query"], staging["query_rope"]), dim=-1).float()
    key_nope = _paged_to_tnd(staging["key"], staging["block_table"])
    key_rope = _paged_to_tnd(staging["key_rope"], staging["block_table"])
    key = torch.cat((key_nope, key_rope), dim=-1).reshape(-1, 1, 576).float()
    scores = torch.einsum("qhd,shd->qhs", query, key) * staging["scale_value"]
    expected_lse = torch.logsumexp(scores, dim=-1)
    actual_lse = (graph_max + torch.log(graph_sum)).squeeze(0)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-3, atol=1e-3)
