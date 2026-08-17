import torch

from sgl_kernel_npu.dspark.top1 import select_global_top1_npu


def test_tp_top1_selector_matches_argmax_under_graph_replay():
    device = torch.device("npu")
    candidates = torch.tensor(
        [
            [[1.0, 8.0], [2.0, 7.0], [2.0, 3.0], [0.0, 1.0]],
            [[-1.0, 9.0], [-2.0, 4.0], [-3.0, 5.0], [-4.0, 6.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    expected = torch.tensor([3, 9], dtype=torch.long, device=device)
    actual = select_global_top1_npu(candidates, vocab_size=16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_actual = select_global_top1_npu(candidates, vocab_size=16)
    candidates.copy_(
        torch.tensor(
            [
                [[4.0, 8.0], [3.0, 7.0], [2.0, 3.0], [1.0, 1.0]],
                [[0.0, 9.0], [5.0, 11.0], [5.0, 6.0], [4.0, 2.0]],
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    graph.replay()
    torch.npu.synchronize()
    expected_after_replay = torch.tensor([8, 6], dtype=torch.long, device=device)
    torch.testing.assert_close(graph_actual, expected_after_replay, rtol=0, atol=0)
