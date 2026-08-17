import math

import pytest
import sgl_kernel_npu  # noqa: F401  Registers torch.ops.sgl_kernel_npu.
import torch
import torch_npu

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")


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
    return torch.ops.sgl_kernel_npu.npu_sparse_flash_attention(
        **inputs,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=return_softmax_lse,
    )


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

    output, softmax_max, softmax_sum = (
        torch.ops.sgl_kernel_npu.npu_sparse_flash_attention(
            **inputs,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
    )
    torch.npu.synchronize()

    query = torch.cat((inputs["query"], inputs["query_rope"]), dim=-1).float()
    key = torch.cat((inputs["key"], inputs["key_rope"]), dim=-1).float()
    scores = torch.einsum("qhd,shd->qhs", query, key) * inputs["scale_value"]
    expected_lse = torch.logsumexp(scores, dim=-1)
    actual_lse = (softmax_max + torch.log(softmax_sum)).squeeze(0)

    torch.testing.assert_close(output, _dense_reference(inputs), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-3, atol=1e-3)


def test_sparse_flash_attention_meta_exposes_dcp_lse_shapes():
    query = torch.empty((2, 8, 512), dtype=torch.bfloat16, device="meta")
    key = torch.empty((1, 128, 1, 512), dtype=torch.bfloat16, device="meta")
    sparse_indices = torch.empty((2, 1, 2048), dtype=torch.int32, device="meta")
    output, softmax_max, softmax_sum = (
        torch.ops.sgl_kernel_npu.npu_sparse_flash_attention(
            query,
            key,
            key,
            sparse_indices,
            1.0 / math.sqrt(576),
            layout_query="TND",
            layout_kv="PA_BSND",
            return_softmax_lse=True,
        )
    )
    assert output.shape == query.shape
    assert softmax_max.shape == (1, 2, 8)
    assert softmax_sum.shape == (1, 2, 8)


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
