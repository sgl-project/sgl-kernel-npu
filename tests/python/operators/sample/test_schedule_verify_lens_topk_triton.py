from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(8, 6)]


@dataclass
class ScheduleConfig:
    gamma: int
    min_verify_len: int = 1
    survival_eps: float = 1e-6
    max_verify_len: int = 0

    def resolved_max_verify_len(self):
        return self.max_verify_len or self.gamma + 1


def generate_data(rows, gamma, dtype, case):
    if case == "ties":
        return torch.ones((rows, gamma), dtype=dtype)
    if case == "strided":
        return torch.rand((rows, gamma * 2), dtype=dtype)[:, ::2]
    if case == "offset":
        return torch.rand(rows * gamma + 1, dtype=dtype)[1:].view(rows, gamma)
    return torch.rand((rows, gamma), dtype=dtype)


@pytest.mark.parametrize(
    "rows,gamma,budget,case",
    [
        # DSpark draft block size 6 and verify length 7; batch is an example.
        pytest.param(8, 6, 16, "random", id="dspark-draft6-verify7-b8"),
        pytest.param(2, 6, 3, "ties", id="equal-probability-order"),
        pytest.param(8, 6, 0, "random", id="zero-budget"),
    ],
)
def test_schedule_verify_lens(rows, gamma, budget, case):
    from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

    torch.manual_seed(42)
    confidence = generate_data(rows, gamma, torch.float32, case)
    kwargs = dict(confidence=confidence, budget=budget, cfg=ScheduleConfig(gamma))
    executor = check_accuracy(schedule_verify_lens_topk_triton, kwargs=kwargs)
    if case == "ties":
        assert torch.equal(executor(), torch.tensor([3, 2], dtype=torch.int32))
    elif budget == 0:
        assert torch.equal(executor(), torch.ones(rows, dtype=torch.int32))
    check_performance(
        executor, enable_perf=(rows, gamma) in perf_shape and budget == 16
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_launchers_use_current_inputs_and_parameters(dtype):
    from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

    cfg = ScheduleConfig(6)
    for budget in (0, 1, 2, 16, 17, 256, 2**40, -1, 1):
        for eps in (0.0, 0.5, 1.0):
            cfg.survival_eps = eps
            # New data and pointers must not reuse the tensors from cache fill.
            confidence = generate_data(8, 6, dtype, "strided")
            actual = schedule_verify_lens_topk_triton(
                confidence=confidence, budget=budget, cfg=cfg
            )
            expected = schedule_verify_lens_topk_triton(
                confidence=confidence.cpu(), budget=budget, cfg=cfg, impl="torch_native"
            )
            assert torch.equal(actual.cpu(), expected)


def test_cached_launchers_graph_replay_and_alignment():
    from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

    # Contiguous, but offset by one FP32 element from an aligned allocation.
    confidence = generate_data(8, 6, torch.float32, "offset")
    for cfg in (ScheduleConfig(6), ScheduleConfig(6, 0, 0.5, 3)):
        kwargs = dict(confidence=confidence, budget=1, cfg=cfg)
        schedule_verify_lens_topk_triton(**kwargs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = schedule_verify_lens_topk_triton(**kwargs)
        for value in (0.0, 1.0):
            confidence.fill_(value)
            graph.replay()
            expected = schedule_verify_lens_topk_triton(
                confidence=confidence.cpu(), budget=1, cfg=cfg, impl="torch_native"
            )
            assert torch.equal(actual.cpu(), expected)
