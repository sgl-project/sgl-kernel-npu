"""Check the distinct mHC V1 and V2 registrations in a single A3 process."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Register both bundled vendors before the shared device fixture runs.
import sgl_kernel_npu


@pytest.mark.parametrize("tokens", [18, 24, 4096])
def test_mhc_pre_sinkhorn_v2_coexists_with_v1(tokens):
    from sgl_kernel_npu.operators.hyper_conn import (
        mhc_pre_sinkhorn,
        mhc_pre_sinkhorn_v2,
    )

    dependencies = getattr(sgl_kernel_npu, "_manifest", {}).get("third_party", ())
    if not any(item["name"] == "sgl_ops_transformer_mhc_v2" for item in dependencies):
        pytest.skip("requires the bundled mHC V2 vendor")

    rng = torch.Generator(device="cpu").manual_seed(tokens)
    with torch.device("cpu"):
        phi = torch.randn(24, 4 * 5120, generator=rng) / (4 * 5120) ** 0.5
        alpha = torch.tensor([0.1, 0.2, 0.3])
        bias = torch.randn(24, generator=rng) * 0.1
        x_cpu = torch.randn(tokens, 4, 5120, dtype=torch.bfloat16, generator=rng) * 0.1
        previous = torch.zeros(tokens, 4)
        previous[:, 0] = 1
    params = (phi, alpha, bias)
    args = (x_cpu.npu(), *(p.npu() for p in params))
    pre_mix = previous.npu()
    v2_kwargs = {"pre_mix": pre_mix, "norm_eps": 1e-20}

    v1_before = mhc_pre_sinkhorn(*args, norm_eps=1e-20)
    eager = mhc_pre_sinkhorn_v2(*args, **v2_kwargs)
    torch.testing.assert_close(eager[0].cpu(), x_cpu[:, 0], atol=0, rtol=0)
    v1_after = mhc_pre_sinkhorn(*args, norm_eps=1e-20)
    for before, after in zip(v1_before, v1_after, strict=True):
        torch.testing.assert_close(before.cpu(), after.cpu(), atol=0, rtol=0)

    if tokens == 4096:
        expected = mhc_pre_sinkhorn_v2(
            x_cpu, *params, pre_mix=previous, norm_eps=1e-20, impl="torch_native"
        )
        for actual, reference in zip(eager[1:], expected[1:], strict=True):
            torch.testing.assert_close(actual.cpu(), reference, atol=2e-4, rtol=2e-4)
        return

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = mhc_pre_sinkhorn_v2(*args, **v2_kwargs)
    for _ in range(2):
        with torch.device("cpu"):
            changed_x = torch.randn(x_cpu.shape, dtype=x_cpu.dtype, generator=rng) * 0.1
            changed_pre = torch.rand(previous.shape, generator=rng)
        args[0].copy_(changed_x)
        pre_mix.copy_(changed_pre)
        graph.replay()
        torch.npu.synchronize()
        expected = mhc_pre_sinkhorn_v2(
            changed_x, *params, pre_mix=changed_pre, norm_eps=1e-20, impl="torch_native"
        )
        torch.testing.assert_close(outputs[0].cpu(), expected[0], atol=1e-5, rtol=1e-2)
        for actual, reference in zip(outputs[1:], expected[1:], strict=True):
            torch.testing.assert_close(actual.cpu(), reference, atol=2e-4, rtol=2e-4)
