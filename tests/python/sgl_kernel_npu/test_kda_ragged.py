import torch

from sgl_kernel_npu.fla.kda_ragged import (
    gather_kda_verify_output_npu,
    scatter_kda_verify_inputs_npu,
)


def _dense_indices(query_start_loc: torch.Tensor, packed_tokens: int, steps: int):
    positions = torch.arange(packed_tokens, dtype=torch.int32, device="npu")
    slots = torch.searchsorted(query_start_loc[1:], positions, right=True)
    return slots * steps + (positions - query_start_loc[slots]).long()


def test_kda_ragged_io_matches_eager_under_graph_replay():
    device = torch.device("npu")
    steps = 4
    query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
    qkv = torch.randn(4, 17, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, 4, 2, 5, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, 4, 2, dtype=torch.bfloat16, device=device)

    dense_qkv, dense_a, dense_b = scatter_kda_verify_inputs_npu(
        qkv, a, b, query_start_loc, draft_token_num=steps
    )
    expected_qkv = torch.zeros(8, 17, dtype=qkv.dtype, device=device)
    expected_a = torch.zeros(1, 8, 2, 5, dtype=a.dtype, device=device)
    expected_b = torch.zeros(1, 8, 2, dtype=b.dtype, device=device)
    expected_qkv[[0, 1, 2, 4]] = qkv
    expected_a[:, [0, 1, 2, 4]] = a
    expected_b[:, [0, 1, 2, 4]] = b
    torch.testing.assert_close(dense_qkv, expected_qkv, rtol=0, atol=0)
    torch.testing.assert_close(dense_a, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(dense_b, expected_b, rtol=0, atol=0)

    dense_output = torch.randn(1, 8, 2, 5, dtype=torch.bfloat16, device=device)
    indices = _dense_indices(query_start_loc, qkv.shape[0], steps)
    packed_output = gather_kda_verify_output_npu(dense_output, indices)
    torch.testing.assert_close(
        packed_output, dense_output[:, [0, 1, 2, 4]], rtol=0, atol=0
    )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_qkv, graph_a, graph_b = scatter_kda_verify_inputs_npu(
            qkv, a, b, query_start_loc, draft_token_num=steps
        )
    query_start_loc.copy_(
        torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    )
    qkv.copy_(torch.randn_like(qkv))
    a.copy_(torch.randn_like(a))
    b.copy_(torch.randn_like(b))
    graph.replay()
    torch.npu.synchronize()
    replay_qkv = torch.zeros_like(expected_qkv)
    replay_a = torch.zeros_like(expected_a)
    replay_b = torch.zeros_like(expected_b)
    replay_qkv[[0, 1, 4, 5]] = qkv
    replay_a[:, [0, 1, 4, 5]] = a
    replay_b[:, [0, 1, 4, 5]] = b
    torch.testing.assert_close(graph_qkv, replay_qkv, rtol=0, atol=0)
    torch.testing.assert_close(graph_a, replay_a, rtol=0, atol=0)
    torch.testing.assert_close(graph_b, replay_b, rtol=0, atol=0)

    gather_graph = torch.npu.NPUGraph()
    with torch.npu.graph(gather_graph):
        graph_indices = _dense_indices(query_start_loc, qkv.shape[0], steps)
        graph_packed = gather_kda_verify_output_npu(dense_output, graph_indices)
    query_start_loc.copy_(
        torch.tensor([0, 3, 4], dtype=torch.int32, device=device)
    )
    dense_output.copy_(torch.randn_like(dense_output))
    gather_graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        graph_packed, dense_output[:, [0, 1, 2, 4]], rtol=0, atol=0
    )
