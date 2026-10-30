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

    def resolved_max_verify_len(self):
        return self.gamma + 1


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
