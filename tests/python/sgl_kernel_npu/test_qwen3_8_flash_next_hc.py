"""Model-scoped HC contracts and CPU FP64 numerical regression checks.

The ordinary FP32 norm has known large-row failures at the unchanged numerical
gate. Keep those failures visible; graph/eager agreement is a separate check.
"""
import pytest
import torch
import torch_npu
from sgl_kernel_npu.qwen3_8_flash_next import hc

pytestmark = pytest.mark.skipif(not torch_npu.npu.is_available(), reason="NPU is required")
OPS = ("grouped_norm", "mix", "combine")
TOLERANCES = {
    "grouped_norm": dict(atol=5e-3, rtol=5e-3),
    "mix": dict(atol=5e-3, rtol=1e-2),
    "combine": dict(atol=5e-3, rtol=1e-2),
}


def inputs(op, rows):
    generator = torch.Generator(device="cpu").manual_seed(73)

    def rand(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    x = rand((rows, 10240))
    if op == "grouped_norm":
        return x, rand((10240,), 0.1), 2560, 1e-6
    if op == "mix":
        return x, rand((320, 10240), 0.02), rand((10240, 320), 0.02), 4, 2560
    return rand((rows, 2560)), x, rand((rows, 10240)), rand((4, 10240), 0.02), 4, 2560


def reference(op, args):
    """CPU FP64, with BF16 rounding only at the public output boundary."""
    tensors = [v.detach().cpu() for v in args if isinstance(v, torch.Tensor)]
    rows = tensors[0].shape[0]
    width = 2560 if op == "mix" else 10240
    result = torch.empty((rows, width), dtype=torch.bfloat16)
    weights = [v.double() for v in (tensors[1:] if op != "combine" else tensors[3:])]
    for start in range(0, rows, 64):
        stop = min(rows, start + 64)
        x = tensors[0][start:stop].double()
        if op == "grouped_norm":
            grouped = x.reshape(-1, 4, 2560)
            square_mean = (grouped * grouped).sum(-1, keepdim=True) / 2560
            y = (grouped / torch.sqrt(square_mean + args[3])).flatten(1) * (1 + weights[0])
        elif op == "mix":
            low = x @ weights[0].T / 4
            hidden = low / (1 + torch.exp(-low))
            gates = 1 / (1 + torch.exp(-(hidden @ weights[1].T)))
            y = (gates.reshape(-1, 4, 2560) * x.reshape(-1, 4, 2560)).sum(1) / 4
        else:
            residual, normed = [v[start:stop].double() for v in tensors[1:3]]
            logits = normed @ weights[0].T / 4
            gate = 2 / (1 + torch.exp(-logits))
            y = (residual.reshape(-1, 4, 2560) + gate[:, :, None] * x[:, None, :]).flatten(1)
        result[start:stop] = y.to(torch.bfloat16)
    return result


def on_device(args):
    return tuple(v.to("npu") if isinstance(v, torch.Tensor) else v for v in args)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("rows", [0, 1, 32, 33, 4096, 8193])
def test_model_fp64(op, rows):
    cpu = inputs(op, rows)
    args = on_device(cpu)
    saved = [v.clone() for v in args if isinstance(v, torch.Tensor)]
    out = getattr(hc, op)(*args)
    assert out.dtype == torch.bfloat16 and out.device == args[0].device
    assert out.is_contiguous()
    for value, original in zip((v for v in args if isinstance(v, torch.Tensor)), saved):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), reference(op, cpu), **TOLERANCES[op])


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("bad", ["cpu", "dtype", "stride", "shape", "scalar", "weight_dtype"])
def test_reject_unsupported_metadata(op, bad):
    args = list(on_device(inputs(op, 1)))
    if bad == "cpu":
        args[0] = args[0].cpu()
    elif bad == "dtype":
        args[0] = args[0].float()
    elif bad == "stride":
        args[0] = torch.stack((args[0], args[0]), -1)[..., 0]
    elif bad == "shape":
        args[0] = args[0][:, :-1].contiguous()
    elif bad == "weight_dtype":
        slot = 3 if op == "combine" else 1
        args[slot] = args[slot].float()
    elif op == "grouped_norm":
        args[2] = 1280
    else:
        args[-2] = 5
    with pytest.raises(ValueError):
        getattr(hc, op)(*args)


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), True])
def test_reject_invalid_epsilon(eps):
    x, w, group, _ = on_device(inputs("grouped_norm", 0))
    with pytest.raises(ValueError):
        hc.grouped_norm(x, w, group, eps)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("rows", [0, 33, 8192, 8193, 16384])
def test_resource_dispatch(monkeypatch, op, rows):
    args = on_device(inputs(op, rows))
    calls = []
    sentinel = object()
    names = ("grouped_norm",) if op == "grouped_norm" else (op, op + "_chunked")
    for name in names:
        def stub(*args, _name=name):
            calls.append(_name)
            return sentinel
        monkeypatch.setattr(hc.core, name, stub)
    out = getattr(hc, op)(*args)
    if rows == 0:
        assert out.shape == (0, 2560 if op == "mix" else 10240)
        assert not calls
    else:
        assert out is sentinel
        expected = op if op == "grouped_norm" or rows <= 8192 else op + "_chunked"
        assert calls == [expected]


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("rows", [33, 8193])
def test_graph_updates_match_eager(op, rows):
    """Replay correctness only; the independent numerical gate is above."""
    args = on_device(inputs(op, rows))
    tensors = [v for v in args if isinstance(v, torch.Tensor)]
    pointers = [v.data_ptr() for v in tensors]
    fn = getattr(hc, op)
    for _ in range(3):
        fn(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fn(*args)
    for slot in range(-1, len(tensors)):
        if slot >= 0:
            tensors[slot].mul_(-0.5).add_(0.03125)
        graph.replay()
        torch.npu.synchronize()
        assert [v.data_ptr() for v in tensors] == pointers
        torch.testing.assert_close(out, fn(*args), atol=0, rtol=0)


def test_combine_fp32_cancellation():
    # Do not require exact zero from the obsolete BF16 intermediate rounding.
    args = list(on_device(inputs("combine", 1)))
    block, _, normed, weight = args[:4]
    gate = 2 * torch.sigmoid(normed.cpu().double() @ weight.cpu().double().T / 4)
    args[1] = (-(block.cpu().double()[:, None, :] * gate[:, :, None])).flatten(1).to(
        device=block.device, dtype=torch.bfloat16
    )
    torch.testing.assert_close(hc.combine(*args).cpu(), reference("combine", args), **TOLERANCES["combine"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
