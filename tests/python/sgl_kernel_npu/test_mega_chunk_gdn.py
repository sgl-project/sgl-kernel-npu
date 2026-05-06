import pytest
import torch
import torch.nn.functional as F

from sgl_kernel_npu.fla.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_native,
)


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float().cpu() - expected.float()).abs()
    max_abs = diff.max().item()
    if max_abs <= 1e-2:
        return

    rmse = torch.sqrt((diff.flatten() ** 2).mean()).item()
    base = torch.sqrt((expected.float().flatten() ** 2).mean()).item()
    ratio = rmse / max(base, 1e-8)
    assert ratio < 0.05, f"{name} max_abs={max_abs:.6f} rmse_ratio={ratio:.6f}"


def _native_reference(q, k, v, g, beta, cu_seqlens):
    if cu_seqlens is None:
        return chunk_gated_delta_rule_native(
            query=q,
            key=k,
            value=v,
            g=g,
            beta=beta,
            chunk_size=128,
            initial_state=None,
            output_final_state=True,
        )

    outs = []
    states = []
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        out, state = chunk_gated_delta_rule_native(
            query=q[:, start:end],
            key=k[:, start:end],
            value=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            chunk_size=128,
            initial_state=None,
            output_final_state=True,
        )
        outs.append(out)
        states.append(state)
    return torch.cat(outs, dim=1), torch.cat(states, dim=0)


@pytest.mark.parametrize(
    ("total_tokens", "cu_list"),
    [
        (128, None),
        (256, [0, 64, 256]),
    ],
)
def test_mega_chunk_gdn_e2e(monkeypatch, total_tokens, cu_list):
    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("mega_chunk_gdn op is not registered")

    monkeypatch.setenv("SGL_KERNEL_NPU_GDN_FORCE_MEGA", "1")

    torch.manual_seed(0)
    device = torch.device("npu")
    H = 16
    D = 128

    q_cpu = F.normalize(torch.randn(1, total_tokens, H, D), p=2, dim=-1).to(torch.float16)
    k_cpu = F.normalize(torch.randn(1, total_tokens, H, D), p=2, dim=-1).to(torch.float16)
    v_cpu = torch.randn(1, total_tokens, H, D, dtype=torch.float16)
    g_cpu = F.logsigmoid(torch.randn(1, total_tokens, H, dtype=torch.float32))
    beta_cpu = torch.rand(1, total_tokens, H, dtype=torch.float16)

    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    g = g_cpu.to(device)
    beta = beta_cpu.to(device)
    cu = None if cu_list is None else torch.tensor(cu_list, dtype=torch.long, device=device)

    # _, actual, _, actual_state, _, _, _ = chunk_gated_delta_rule_fwd(
    #     q=q,
    #     k=k,
    #     v=v,
    #     g=g,
    #     beta=beta,
    #     scale=D**-0.5,
    #     initial_state=None,
    #     output_final_state=True,
    #     cu_seqlens=cu,
    # )
    # torch.npu.synchronize()

    expected, expected_state = _native_reference(q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu, cu_list)
    _assert_close("output", actual, expected)
    _assert_close("final_state", actual_state, expected_state)
