import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sgl_kernel_npu"
    / "sgl_kernel_npu"
    / "norm"
    / "gemma_rmsnorm.py"
)


class OffsetWeight:
    def __radd__(self, value):
        return (value, self)


def load_gemma_module(monkeypatch, use_native, torch_npu):
    torch = ModuleType("torch")
    torch.Tensor = object
    torch.ops = SimpleNamespace(
        npu=SimpleNamespace(sgl_kernel_npu_use_native_gemma_rms_norm=lambda: use_native)
    )

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "sgl_kernel_npu", ModuleType("sgl_kernel_npu"))
    monkeypatch.setitem(
        sys.modules, "sgl_kernel_npu.norm", ModuleType("sgl_kernel_npu.norm")
    )
    module_name = "sgl_kernel_npu.norm.gemma_rmsnorm"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_native_provider_uses_torch_npu_gemma_operators(monkeypatch):
    calls = []

    def npu_gemma_rms_norm(input, weight, eps):
        calls.append(("plain", input, weight, eps))
        return "plain-output", "rstd"

    torch_npu = SimpleNamespace(
        npu_gemma_rms_norm=npu_gemma_rms_norm,
    )
    module = load_gemma_module(
        monkeypatch,
        use_native=True,
        torch_npu=torch_npu,
    )
    weight = OffsetWeight()

    assert module.npu_gemma_rms_norm("input", weight, 1e-5) == (
        "plain-output",
        "rstd",
    )
    assert calls == [("plain", "input", weight, 1e-5)]


def test_aclnn_provider_uses_standard_rms_norm_operators(monkeypatch):
    calls = []

    def npu_rms_norm(input, weight, eps):
        calls.append(("plain", input, weight, eps))
        return "plain-output", "rstd"

    module = load_gemma_module(
        monkeypatch,
        use_native=False,
        torch_npu=SimpleNamespace(
            npu_rms_norm=npu_rms_norm,
        ),
    )
    weight = OffsetWeight()

    assert module.npu_gemma_rms_norm("input", weight, 1e-5) == (
        "plain-output",
        "rstd",
    )
    assert calls == [("plain", "input", (1.0, weight), 1e-5)]
