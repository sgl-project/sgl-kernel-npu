import pytest
import torch
import torch.nn.functional as F

from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_native
from sgl_kernel_npu.fla.mega_chunk_gdn import run_mega_chunk_gdn


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")

SUPPORTED_HEAD_CONFIGS = [
    pytest.param(16, 16, id="H16-Hg16"),
    pytest.param(32, 16, id="H32-Hg16"),
    pytest.param(48, 16, id="H48-Hg16"),
    pytest.param(64, 16, id="H64-Hg16"),
    pytest.param(16, 8, id="H16-Hg8"),
    pytest.param(32, 8, id="H32-Hg8"),
    pytest.param(16, 4, id="H16-Hg4"),
]


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float().cpu() - expected.float()).abs()
    max_abs = diff.max().item()
    if max_abs <= 1e-2:
        return

    rmse = torch.sqrt((diff.flatten() ** 2).mean()).item()
    base = torch.sqrt((expected.float().flatten() ** 2).mean()).item()
    ratio = rmse / max(base, 1e-8)
    assert ratio < 0.05, f"{name} max_abs={max_abs:.6f} rmse_ratio={ratio:.6f}"


def _native_reference(q, k, v, g, beta, cu_seqlens, initial_state=None, output_final_state=False):
    if q.shape[2] != v.shape[2]:
        assert v.shape[2] % q.shape[2] == 0
        group_size = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(group_size, dim=2)
        k = k.repeat_interleave(group_size, dim=2)

    if cu_seqlens is None:
        out, final_state = chunk_gated_delta_rule_native(
            query=q,
            key=k,
            value=v,
            g=g,
            beta=beta,
            chunk_size=128,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return out, final_state

    outs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        cur_initial_state = (
            None if initial_state is None else initial_state[seq_idx : seq_idx + 1]
        )
        out, final_state = chunk_gated_delta_rule_native(
            query=q[:, start:end],
            key=k[:, start:end],
            value=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            chunk_size=128,
            initial_state=cur_initial_state,
            output_final_state=output_final_state,
        )
        outs.append(out)
        if output_final_state:
            final_states.append(final_state)
    return torch.cat(outs, dim=1), (
        torch.cat(final_states, dim=0) if output_final_state else None
    )


@pytest.mark.parametrize(
    ("total_tokens", "cu_list"),
    [
        (129, None),
        (256, [0, 96, 128, 256]),
        (2560, [0, 96, 128, 2560]),
    ],
)
@pytest.mark.parametrize("num_value_heads", [16, 32, 48, 64])
def test_mega_chunk_gdn_e2e(total_tokens, cu_list, num_value_heads):
    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("mega_chunk_gdn op is not registered")

    torch.manual_seed(0)
    device = torch.device("npu")
    Hg = 16
    H = num_value_heads
    D = 128

    q_cpu = F.normalize(torch.randn(1, total_tokens, Hg, D), p=2, dim=-1).to(torch.float16)
    k_cpu = F.normalize(torch.randn(1, total_tokens, Hg, D), p=2, dim=-1).to(torch.float16)
    v_cpu = torch.randn(1, total_tokens, H, D, dtype=torch.float16)
    g_cpu = F.logsigmoid(torch.randn(1, total_tokens, H, dtype=torch.float32))
    beta_cpu = torch.rand(1, total_tokens, H, dtype=torch.float16)

    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    g = g_cpu.to(device)
    beta = beta_cpu.to(device)
    cu = None if cu_list is None else torch.tensor(cu_list, dtype=torch.long, device=device)
    scale = D**-0.5

    _, actual, _, _, _, _, _ = run_mega_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu,
    )
    torch.npu.synchronize()

    expected, _ = _native_reference(q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu, cu_list)
    _assert_close("mega_vs_native", actual, expected)


@pytest.mark.parametrize(
    ("total_tokens", "cu_list"),
    [
        (129, None),
        (256, [0, 96, 128, 256]),
    ],
)
@pytest.mark.parametrize("num_value_heads", [16, 32])
@pytest.mark.parametrize("state_kind", ["zero", "random"])
def test_mega_chunk_gdn_initial_state(total_tokens, cu_list, num_value_heads, state_kind):
    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("mega_chunk_gdn op is not registered")

    torch.manual_seed(1)
    device = torch.device("npu")
    Hg = 16
    H = num_value_heads
    D = 128
    num_sequences = 1 if cu_list is None else len(cu_list) - 1

    q_cpu = F.normalize(torch.randn(1, total_tokens, Hg, D), p=2, dim=-1).to(torch.float16)
    k_cpu = F.normalize(torch.randn(1, total_tokens, Hg, D), p=2, dim=-1).to(torch.float16)
    v_cpu = torch.randn(1, total_tokens, H, D, dtype=torch.float16)
    g_cpu = F.logsigmoid(torch.randn(1, total_tokens, H, dtype=torch.float32))
    beta_cpu = torch.rand(1, total_tokens, H, dtype=torch.float16)
    if state_kind == "zero":
        h0_cpu = torch.zeros(num_sequences, H, D, D, dtype=torch.float16)
    else:
        h0_cpu = (0.1 * torch.randn(num_sequences, H, D, D)).to(torch.float16)

    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    g = g_cpu.to(device)
    beta = beta_cpu.to(device)
    h0 = h0_cpu.to(device)
    cu = None if cu_list is None else torch.tensor(cu_list, dtype=torch.long, device=device)
    scale = D**-0.5

    _, actual, _, actual_final_state, _, _, _ = run_mega_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    torch.npu.synchronize()

    expected, expected_final_state = _native_reference(
        q_cpu,
        k_cpu,
        v_cpu,
        g_cpu,
        beta_cpu,
        cu_list,
        initial_state=h0_cpu,
        output_final_state=True,
    )
    _assert_close(f"mega_vs_native_{state_kind}", actual, expected)
    _assert_close(
        f"mega_final_state_vs_native_{state_kind}",
        actual_final_state,
        expected_final_state,
    )


@pytest.mark.parametrize(("num_value_heads", "num_key_heads"), SUPPORTED_HEAD_CONFIGS)
def test_mega_chunk_gdn_all_supported_head_configs(num_value_heads, num_key_heads):
    if not hasattr(torch.ops.npu, "mega_chunk_gdn"):
        pytest.skip("mega_chunk_gdn op is not registered")

    torch.manual_seed(2)
    device = torch.device("npu")
    total_tokens = 129
    cu_list = [0, 64, total_tokens]
    num_sequences = len(cu_list) - 1
    H = num_value_heads
    Hg = num_key_heads
    D = 128

    q_cpu = F.normalize(torch.randn(1, total_tokens, Hg, D), p=2, dim=-1).to(torch.float16)
    k_cpu = F.normalize(torch.randn(1, total_tokens, Hg, D), p=2, dim=-1).to(torch.float16)
    v_cpu = torch.randn(1, total_tokens, H, D, dtype=torch.float16)
    g_cpu = F.logsigmoid(torch.randn(1, total_tokens, H, dtype=torch.float32))
    beta_cpu = torch.rand(1, total_tokens, H, dtype=torch.float16)
    h0_cpu = (0.05 * torch.randn(num_sequences, H, D, D)).to(torch.float16)

    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    g = g_cpu.to(device)
    beta = beta_cpu.to(device)
    h0 = h0_cpu.to(device)
    cu = torch.tensor(cu_list, dtype=torch.long, device=device)
    scale = D**-0.5

    _, actual, _, actual_final_state, _, _, _ = run_mega_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    torch.npu.synchronize()

    expected, expected_final_state = _native_reference(
        q_cpu,
        k_cpu,
        v_cpu,
        g_cpu,
        beta_cpu,
        cu_list,
        initial_state=h0_cpu,
        output_final_state=True,
    )
    _assert_close(f"mega_all_configs_H{H}_Hg{Hg}", actual, expected)
    _assert_close(
        f"mega_all_configs_final_state_H{H}_Hg{Hg}",
        actual_final_state,
        expected_final_state,
    )
