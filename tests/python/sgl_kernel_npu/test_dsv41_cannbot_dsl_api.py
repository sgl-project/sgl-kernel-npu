"""CPU-only checks for the vendored DeepSeek V4.1 DSL entry points."""

import ast
import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import pytest
from setuptools import find_namespace_packages

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "python" / "sgl_kernel_npu"
PACKAGE = PACKAGE_ROOT / "sgl_kernel_npu"
IMPL = PACKAGE / "implementation" / "cannbot-dsl" / "attention"
SOURCE_NAMES = (
    "indexer_prologue_k",
    "mixed_quant_sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "quant_sparse_lightning_indexer",
    "quant_sparse_lightning_indexer_metadata",
)

ATTENTION_OPS = {
    "mixed_quant_sparse_flash_mla",
    "mixed_quant_sparse_flash_mla_metadata",
}


def test_dsl_sources_and_license_are_in_distribution_package():
    packages = find_namespace_packages(where=str(PACKAGE_ROOT))
    assert "sgl_kernel_npu.implementation.cannbot-dsl.attention" in packages
    license_path = PACKAGE / "implementation" / "cannbot-dsl" / "LICENSE.cannbot-dsl"
    assert "CANN Open Software License Agreement Version 2.0" in license_path.read_text(
        encoding="utf-8"
    )


def test_mqsmla_relative_metadata_module_is_in_same_package():
    tree = ast.parse(
        (IMPL / "mixed_quant_sparse_flash_mla.py").read_text(encoding="utf-8")
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "mixed_quant_sparse_flash_mla_metadata"
        for node in ast.walk(tree)
    )


def _load_public_operator(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    class Generation(Enum):
        ARCH35 = "arch35"

    registered = {}

    def register_impl(op_name, impl, *, supported_generations):
        assert supported_generations == {Generation.ARCH35}

        def decorate(fn):
            registered[(op_name, impl)] = fn
            return fn

        return decorate

    registry = types.ModuleType("sgl_kernel_npu.operators.utils.registry")
    registry.register_impl = register_impl
    registry.register_operator = lambda _: lambda fn: fn
    registry.get_impl_by_name = lambda op, impl: registered[(op, impl)]
    generation = types.ModuleType("sgl_kernel_npu.operators.utils.generation")
    generation.Generation = Generation

    for package in (
        "sgl_kernel_npu",
        "sgl_kernel_npu.operators",
        "sgl_kernel_npu.operators.utils",
        "sgl_kernel_npu.operators.indexer",
        "sgl_kernel_npu.operators.attention",
    ):
        module = types.ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, generation.__name__, generation)

    helper_name = "sgl_kernel_npu.operators.utils.cannbot_dsl"
    helper_path = PACKAGE / "operators" / "utils" / "cannbot_dsl.py"
    helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    monkeypatch.setitem(sys.modules, helper_name, helper)
    helper_spec.loader.exec_module(helper)
    monkeypatch.setattr(helper, "initialize_backend", lambda: None)

    category = "attention" if name in ATTENTION_OPS else "indexer"
    module_name = f"sgl_kernel_npu.operators.{category}.{name}"
    module_path = PACKAGE / "operators" / category / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return getattr(module, name), registered, helper, category


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_public_operator_preserves_upstream_signature_and_lazy_dispatch(
    monkeypatch, name
):
    category = "attention" if name in ATTENTION_OPS else "indexer"
    source_tree = ast.parse((IMPL / f"{name}.py").read_text(encoding="utf-8"))
    source_fn = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    public_path = PACKAGE / "operators" / category / f"{name}.py"
    public_tree = ast.parse(public_path.read_text(encoding="utf-8"))
    public_fn = next(
        node
        for node in public_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    assert [arg.arg for arg in public_fn.args.args] == [
        arg.arg for arg in source_fn.args.args
    ]
    assert [arg.arg for arg in public_fn.args.kwonlyargs] == [
        arg.arg for arg in source_fn.args.kwonlyargs
    ] + ["impl"]
    assert [ast.unparse(value) for value in public_fn.args.defaults] == [
        ast.unparse(value) for value in source_fn.args.defaults
    ]
    assert [
        ast.unparse(value) if value else None
        for value in public_fn.args.kw_defaults[:-1]
    ] == [ast.unparse(value) if value else None for value in source_fn.args.kw_defaults]

    with monkeypatch.context() as scoped:
        public, registered, helper, category = _load_public_operator(scoped, name)
        calls = []
        marker = object()

        def fake_vendor(*args, **kwargs):
            calls.append((args, kwargs))
            return marker

        imports = []

        def fake_import(module_name):
            imports.append(module_name)
            return types.SimpleNamespace(**{name: fake_vendor})

        device_checks = []
        public_module = sys.modules[f"sgl_kernel_npu.operators.{category}.{name}"]

        @contextmanager
        def fake_device(*values, **named):
            device_checks.append((values, named))
            yield

        scoped.setattr(
            public_module,
            "ascend950_device",
            fake_device,
            raising=False,
        )
        scoped.setattr(public_module, "require_ascend950", lambda *a, **kw: 0)
        scoped.setattr(helper.importlib, "import_module", fake_import)
        signature = inspect.signature(public)
        args = [
            object()
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        kwargs = {
            p.name: object()
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert not imports
        assert public(*args, **kwargs) is marker
        assert public(*args, **kwargs, impl="cannbot_dsl") is marker
        assert len(device_checks) == 2
        assert imports == [
            f"sgl_kernel_npu.implementation.cannbot-dsl.attention.{name}"
        ]
        op_name = f"{category}.{name}"
        assert registered[(op_name, "auto")] is registered[(op_name, "cannbot_dsl")]
        expected = signature.bind(*args, **kwargs)
        expected.apply_defaults()
        expected.arguments.pop("impl")
        for backend_args, backend_kwargs in calls:
            actual = signature.bind(*backend_args, **backend_kwargs)
            actual.apply_defaults()
            actual.arguments.pop("impl")
            assert actual.arguments == expected.arguments


def test_a3_is_rejected_before_loading_dsl(monkeypatch):
    helper_name = "sgl_kernel_npu.operators.utils.cannbot_dsl"
    helper_path = PACKAGE / "operators" / "utils" / "cannbot_dsl.py"
    spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    torch_stub = types.ModuleType("torch")
    torch_stub.npu = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_name=lambda _: "Ascend910_9362",
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    tensor = types.SimpleNamespace(device=types.SimpleNamespace(type="npu", index=0))
    with pytest.raises(RuntimeError, match="Ascend 950"):
        helper.require_ascend950(tensor)

    torch_stub.npu.get_device_name = lambda _: "Ascend950PR_9599"
    helper.require_ascend950(tensor)


def test_mixed_npu_devices_are_rejected_before_dsl_launch(monkeypatch):
    helper_name = "sgl_kernel_npu.operators.utils.cannbot_dsl"
    helper_path = PACKAGE / "operators" / "utils" / "cannbot_dsl.py"
    spec = importlib.util.spec_from_file_location(helper_name, helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch_stub = types.ModuleType("torch")
    torch_stub.npu = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_name=lambda _: "Ascend950PR_9599",
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    first = types.SimpleNamespace(device=types.SimpleNamespace(type="npu", index=0))
    second = types.SimpleNamespace(device=types.SimpleNamespace(type="npu", index=1))
    with pytest.raises(RuntimeError, match="same NPU device"):
        helper.require_ascend950(first, second)


@pytest.mark.parametrize(
    "name", ("indexer_prologue_k", "quant_lightning_indexer_metadata")
)
def test_public_api_reports_ascend950_requirement_before_arch_registry(
    monkeypatch, name
):
    with monkeypatch.context() as scoped:
        public, _, _, category = _load_public_operator(scoped, name)
        module = sys.modules[f"sgl_kernel_npu.operators.{category}.{name}"]

        def reject_device(*args, **kwargs):
            raise RuntimeError("Ascend 950 is required; found Ascend910_9362")

        def reject_generation(*args, **kwargs):
            raise RuntimeError("arch22 unsupported")

        scoped.setattr(module, "require_ascend950", reject_device)
        scoped.setattr(module, "get_impl_by_name", reject_generation)
        signature = inspect.signature(public)
        required_args = [
            object()
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        required_kwargs = {
            p.name: object()
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        with pytest.raises(RuntimeError, match="Ascend 950"):
            public(*required_args, **required_kwargs)


@pytest.mark.parametrize(
    "name",
    ("quant_lightning_indexer_metadata", "quant_sparse_lightning_indexer_metadata"),
)
def test_metadata_uses_nondefault_input_device_context(monkeypatch, name):
    with monkeypatch.context() as scoped:
        public, _, helper, category = _load_public_operator(scoped, name)
        current = {"index": 0}

        @contextmanager
        def device(index):
            previous = current["index"]
            current["index"] = index
            try:
                yield
            finally:
                current["index"] = previous

        torch_stub = types.ModuleType("torch")
        torch_stub.npu = types.SimpleNamespace(
            current_device=lambda: current["index"],
            get_device_name=lambda _: "Ascend950PR_9599",
            device=device,
        )
        scoped.setitem(sys.modules, "torch", torch_stub)
        observed = []

        def fake_vendor(*args, **kwargs):
            observed.append(current["index"])

        scoped.setattr(
            helper.importlib,
            "import_module",
            lambda _: types.SimpleNamespace(**{name: fake_vendor}),
        )
        tensor = types.SimpleNamespace(
            device=types.SimpleNamespace(type="npu", index=1)
        )
        if name == "quant_lightning_indexer_metadata":
            public(
                cu_seqlens_k=tensor,
                num_heads_q=32,
                num_heads_k=1,
                head_dim=128,
                topk=512,
            )
        else:
            public(
                tensor,
                num_heads_q=32,
                num_heads_k=1,
                head_dim=128,
                topk=512,
                quant_mode=1,
                candidate_block_size=8,
                layout_k="PA_BBND",
            )
        assert observed == [1], (category, name)
        assert current["index"] == 0


@pytest.mark.parametrize("category", ("attention", "indexer"))
def test_public_packages_export_dsv41_operators(category):
    names = {
        name
        for name in SOURCE_NAMES
        if (name in ATTENTION_OPS) == (category == "attention")
    }
    tree = ast.parse(
        (PACKAGE / "operators" / category / "__init__.py").read_text(encoding="utf-8")
    )
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 1
        for alias in node.names
    }
    assert names <= set(exports)
    assert names <= imported


def test_wheel_package_data_includes_cannbot_license():
    setup = ast.parse((PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node.value
        for node in setup.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "setup"
    )
    package_data = ast.literal_eval(
        next(
            keyword.value
            for keyword in setup_call.keywords
            if keyword.arg == "package_data"
        )
    )
    assets = package_data["sgl_kernel_npu.implementation.cannbot-dsl"]
    assert "LICENSE.cannbot-dsl" in assets
    assert "SOURCE.md" in assets


def test_wheel_builders_use_expected_license_metadata():
    expected = {
        PACKAGE_ROOT / "setup.py": "BSD 3 License",
        PACKAGE_ROOT / "build_tools/torch_wheel.py": "BSD 3 License",
    }
    for path, expected_license in expected.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        licenses = [
            ast.literal_eval(keyword.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "license"
        ]
        assert licenses == [expected_license]


def test_all_seven_operators_are_documented_and_indexed():
    for language in ("zh", "en"):
        index = (ROOT / "docs/api" / language / "API_list.md").read_text(
            encoding="utf-8"
        )
        for name in SOURCE_NAMES:
            category = "attention" if name in ATTENTION_OPS else "indexer"
            relative = f"operators/{category}/{name}.md"
            assert f"[{name}]({relative})" in index
            document = ROOT / "docs/api" / language / relative
            text = document.read_text(encoding="utf-8")
            assert "arch35" in text
            assert "cannbot_dsl" in text
            assert "Ascend910_9362" in text
            assert "71ee7fd" in text
            assert name in text
