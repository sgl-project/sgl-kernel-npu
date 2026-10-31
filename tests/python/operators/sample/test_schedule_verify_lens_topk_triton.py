from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@dataclass
class ScheduleConfig:
    gamma: int
    min_verify_len: int = 1
    survival_eps: float = 1e-6
    max_verify_len: int = 0

    def resolved_max_verify_len(self):
        return self.max_verify_len or self.gamma + 1


@pytest.mark.parametrize("case", ["random", "ties", "zero_budget"])
def test_schedule_verify_lens(case):
    from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

    torch.manual_seed(42)
    confidence = torch.rand((8, 6), dtype=torch.float32)
    budget = 16
    if case == "ties":
        confidence = torch.ones((2, 6), dtype=torch.float32)
        budget = 3
    elif case == "zero_budget":
        budget = 0
    kwargs = dict(confidence=confidence, budget=budget, cfg=ScheduleConfig(6))
    executor = check_accuracy(schedule_verify_lens_topk_triton, kwargs=kwargs)
    if case == "ties":
        assert torch.equal(executor(), torch.tensor([3, 2], dtype=torch.int32))
    elif case == "zero_budget":
        assert torch.equal(executor(), torch.ones(8, dtype=torch.int32))
    check_performance(executor, enable_perf=case == "random")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cached_launchers_use_current_inputs_and_parameters(dtype):
    from sgl_kernel_npu.implementation.torch_native.sample.schedule_verify_lens_topk import (
        schedule_verify_lens_topk_triton as reference,
    )
    from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

    cfg = ScheduleConfig(6)
    for budget in (0, 1, 2, 16, 17, 256, 2**40, -1, 1):
        for eps in (0.0, 0.5, 1.0):
            cfg.survival_eps = eps
            # New data and pointers must not reuse the tensors from cache fill.
            confidence = torch.rand((8, 12), dtype=dtype).npu()[:, ::2]
            actual = schedule_verify_lens_topk_triton(
                confidence=confidence, budget=budget, cfg=cfg
            )
            expected = reference(confidence=confidence.cpu(), budget=budget, cfg=cfg)
            assert torch.equal(actual.cpu(), expected)


def test_cached_launchers_graph_replay_and_alignment():
    from sgl_kernel_npu.implementation.torch_native.sample.schedule_verify_lens_topk import (
        schedule_verify_lens_topk_triton as reference,
    )
    from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

    # Contiguous, but offset by one FP32 element from an aligned allocation.
    confidence = torch.rand(49).npu()[1:].view(8, 6)
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
            expected = reference(confidence=confidence.cpu(), budget=1, cfg=cfg)
            assert torch.equal(actual.cpu(), expected)
