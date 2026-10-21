"""Host checks for DSPark API/launch contracts; never execute a device kernel."""

import ast
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"


@pytest.fixture
def launch_modules(monkeypatch):
    calls = []

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                options = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("num_warps", "num_stages")
                }
                inspect.signature(self.fn).bind(*args, **options)
                assert kwargs.get("num_warps") == 1
                assert len(grid) == 1 and 1 <= grid[0] <= 48
                calls.append((self.fn.__name__, grid, options))

            return launch

    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **kw: (
        Kernel(fn) if fn is not None else lambda f: Kernel(f)
    )
    triton.cdiv = lambda n, d: (n + d - 1) // d
    triton.next_power_of_2 = lambda n: 1 << (max(n, 1) - 1).bit_length()
    tl = types.ModuleType("triton.language")
    tl.constexpr = lambda value: value
    triton.language = tl
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    for name, path in [
        ("sgl_kernel_npu", PACKAGE),
        ("sgl_kernel_npu.sample", PACKAGE / "sample"),
        ("sgl_kernel_npu.attention", PACKAGE / "attention"),
        ("sgl_kernel_npu.utils", PACKAGE / "utils"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
    props = types.ModuleType("sgl_kernel_npu.utils.triton_utils")
    props.get_device_properties = lambda: (24, 48)
    monkeypatch.setitem(sys.modules, props.__name__, props)
    before = set(sys.modules)

    def load(name):
        return importlib.import_module("sgl_kernel_npu." + name)

    yield load, calls
    for name in set(sys.modules) - before:
        if name.startswith("sgl_kernel_npu."):
            sys.modules.pop(name, None)


def test_metadata_launch_and_return_contract(launch_modules):
    load, calls = launch_modules
    mod = load("attention.dspark_metadata")
    seq = torch.tensor([1, 5, 7], dtype=torch.int64)
    out = mod.BuildBlockSeqLensCausal.execute(seq_lens=seq, block_size=5, device="cpu")
    assert out.shape == (15,) and out.dtype == torch.int32
    result = mod.ComputeDsparkWindowGather.execute(
        seq_lens_casual=out,
        req_pool_indices_repeated=torch.zeros(15, dtype=torch.int64),
        block_size=5,
        swa_window=128,
    )
    assert result.bs == 3 and result.num_q == 15
    assert result.offsets.shape == (3, 128)
    assert result.invalid.dtype == torch.bool
    assert len(calls) == 2


def test_accept_and_output_never_call_native_argmax(launch_modules, monkeypatch):
    load, calls = launch_modules
    mod = load("sample.dspark_accept")
    window = load("sample.dspark_verify_window")
    monkeypatch.setattr(torch, "argmax", lambda *a, **kw: pytest.fail("native argmax"))
    result = mod.AcceptGreedy.execute(
        candidates=torch.zeros(3, 6, dtype=torch.int64),
        target_logits=torch.empty(18, 129280),
        verify_num_draft_tokens=6,
    )
    assert [x.dtype for x in result] == [torch.int32, torch.int64, torch.int32]
    assert all(x.shape == (3,) for x in result)
    out = window.BuildOutTokens.execute(
        draft_tokens=torch.empty(3, 5, dtype=torch.int64),
        correct_len=result[0],
        bonus=result[1],
        verify_num_draft_tokens=6,
        gamma=5,
    )
    assert out.shape == (3, 6) and out.dtype == torch.int64
    assert [x[0] for x in calls] == [
        "_argmax_kernel",
        "_prefix_kernel",
        "_build_out_tokens_kernel",
    ]


def test_compact_uses_triton_scan_and_preserves_npu_cache_dtype(
    launch_modules, monkeypatch
):
    load, calls = launch_modules
    compact = load("sample.dspark_compact")
    monkeypatch.setattr(torch, "cumsum", lambda *a, **kw: pytest.fail("native cumsum"))
    layout = types.SimpleNamespace(
        verify_lens=torch.tensor([1, 3], dtype=torch.int32), graph_num_tokens=12
    )
    batch = types.SimpleNamespace(
        seq_lens=torch.tensor([2, 7]), req_pool_indices=torch.tensor([0, 1])
    )
    runner = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=torch.empty(2, 32, dtype=torch.int32)
        )
    )
    result = compact.BuildRaggedVerifyWindow.execute(
        batch=batch,
        layout=layout,
        draft_block_ids=torch.empty(2, 5, dtype=torch.int64),
        draft_tokens=torch.empty(2, 5, dtype=torch.int64),
        bs=2,
        device="cpu",
        verify_num_draft_tokens=6,
        model_runner=runner,
    )
    assert result.verify_cache_loc.dtype == torch.int32
    assert (
        result.positions.dtype
        == result.verify_ids.dtype
        == result.window_index.dtype
        == torch.int64
    )
    assert all(getattr(result, f).shape == (12,) for f in result.__struct_fields__)
    assert "_qo_indptr_kernel" in [x[0] for x in calls]


def test_projection_launch_and_dtype(launch_modules, monkeypatch):
    load, calls = launch_modules
    mod = load("sample.dspark_kv_proj")
    monkeypatch.setattr(
        torch.nn.functional, "linear", lambda *a, **kw: pytest.fail("native linear")
    )
    result = mod.CommitKvProj.execute(
        main_x=torch.empty(17, 33, dtype=torch.bfloat16),
        wkv_linears=[
            types.SimpleNamespace(weight=torch.empty(65, 33, dtype=torch.bfloat16))
        ],
    )
    assert len(result) == 1 and result[0].shape == (17, 65)
    assert result[0].dtype == torch.bfloat16
    assert [x[0] for x in calls] == ["_project_kernel"]


def test_schedule_and_ring_launch(launch_modules):
    load, calls = launch_modules
    schedule = load("sample.dspark_schedule")
    cfg = types.SimpleNamespace(
        resolved_max_verify_len=lambda: 6, min_verify_len=1, survival_eps=1e-6
    )
    result = schedule.ScheduleVerifyLensTopk.execute(
        confidence=torch.empty(65, 5), budget=7, cfg=cfg
    )
    assert result.shape == (65,) and result.dtype == torch.int32
    ring = load("attention.dspark_ring")
    out, lengths = ring.BuildDsparkSwaRingIndices.execute(
        seq_lens_casual=torch.empty(15, dtype=torch.int32),
        req_pool_indices_repeated=torch.empty(15, dtype=torch.int64),
        valid_from=torch.empty(3, dtype=torch.int64),
        ring_size=256,
        block_size=5,
        swa_window=128,
        page_index_aligned_size=96,
    )
    assert out.shape == (15, 192) and lengths.shape == (15,)
    assert len(calls) == 4


def test_package_has_no_framework_dependency_or_native_compute_fallback():
    forbidden = {
        "torch.argmax",
        "torch.softmax",
        "torch.cumsum",
        "torch.cumprod",
        "torch.nn.functional.linear",
        "torch.matmul",
        "torch.mm",
        "tl.atomic_add",
    }
    for path in PACKAGE.rglob("dspark*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("sglang"), path
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in forbidden, (path, node.lineno)
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("is_cuda", "item", "is_nonzero"), (
                    path,
                    node.lineno,
                )


def test_persistent_kernels_do_not_use_unsupported_early_exits():
    for path in PACKAGE.rglob("dspark*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and any(
                ast.unparse(d).startswith("triton.jit") for d in fn.decorator_list
            ):
                assert not any(
                    isinstance(n, (ast.Return, ast.Break, ast.Continue))
                    for n in ast.walk(fn)
                ), (
                    path,
                    fn.name,
                )


def test_all_migrated_cuda_cases_are_individual_tests():
    path = ROOT / "tests/python/sgl_kernel_npu/test_dspark_kernel_parity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cases = {
        n.name.removeprefix("_case_")
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_case_")
    }
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TestDsparkKernelParity"
    )
    tests = {
        n.name.removeprefix("test_")
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    assert len(cases) == 20 and cases == tests
