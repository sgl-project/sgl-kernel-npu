import torch
from sgl_kernel_npu.moe.add3 import add3_bf16, add3_bf16_covered


def _eager_add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (a + b) + c


def test_add3_bf16_matches_eager_and_graph_replay():
    torch.manual_seed(0)
    device = torch.device("npu")
    shape = (1, 7168)
    a = torch.randn(shape, dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    c = torch.randn_like(a)

    assert add3_bf16_covered(a, b, c)
    actual = add3_bf16(a, b, c)
    torch.testing.assert_close(actual, _eager_add3(a, b, c), rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replay_actual = add3_bf16(a, b, c)
    a.copy_(torch.randn_like(a))
    b.copy_(torch.randn_like(b))
    c.copy_(torch.randn_like(c))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(replay_actual, _eager_add3(a, b, c), rtol=0, atol=0)
