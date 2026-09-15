import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla import kda_scaled_dot_kkt


class _FakeKernel:
    def __init__(self):
        self.grid = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(**kwargs):
            self.kwargs = kwargs

        return launch


def test_fused_scaled_dot_wrapper_launch_contract(monkeypatch):
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(
        kda_scaled_dot_kkt,
        "_chunk_kda_scaled_dot_kkt_fwd_kernel_128",
        fake_kernel,
    )
    q = torch.empty(1, 65, 2, 128)
    k = torch.empty_like(q)
    gk = torch.empty_like(q, dtype=torch.float32)
    beta = torch.empty(1, 65, 2)

    triangular, query_key = kda_scaled_dot_kkt.chunk_kda_scaled_dot_kkt_fwd_npu(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        scale=128**-0.5,
    )

    assert fake_kernel.grid == (2, 2, 2)
    assert fake_kernel.kwargs["BC"] == 32
    assert fake_kernel.kwargs["BK"] == 128
    assert triangular.shape == (1, 65, 2, 64)
    assert query_key.shape == triangular.shape


def test_fused_scaled_dot_wrapper_rejects_unsupported_key_width():
    q = torch.empty(1, 1, 1, 64)
    k = torch.empty_like(q)
    gk = torch.empty_like(q, dtype=torch.float32)
    beta = torch.empty(1, 1, 1)

    with pytest.raises(ValueError, match="K=128"):
        kda_scaled_dot_kkt.chunk_kda_scaled_dot_kkt_fwd_npu(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            scale=64**-0.5,
        )
