"""Check cross-layer MegaMoe naming without importing or compiling NPU code."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "csrc/zerc_moe/source"
OP = ROOT / "mc2/mega_moe"
OPS = ROOT / "torch_extension/npu_ops_transformer/ops"


def _read(root, path):
    return (root / path).read_text(encoding="utf-8")


def _macro_args(source, macro):
    return re.findall(r"\b" + re.escape(macro) + r"\s*\(\s*([A-Za-z_]\w*)", source)


def test_cann_registration_and_dispatch_use_same_isolated_op():
    definition = _read(OP, "op_host/mega_moe_def.cpp")
    inference = _read(OP, "op_host/mega_moe_infershape.cpp")
    tiling = _read(OP, "op_host/op_tiling/mega_moe_tiling.cpp")
    cmake = _read(OP, "op_host/CMakeLists.txt")
    kernel = _read(OP, "op_kernel/mega_moe.cpp")
    graph = ast.parse(_read(OPS, "graph_convert/graph_convert_mega_moe.py"))

    op_types = _macro_args(definition, "OP_ADD")
    assert op_types == _macro_args(inference, "IMPL_OP_INFERSHAPE")
    assert op_types == _macro_args(tiling, "IMPL_OP_OPTILING")
    assert op_types == ["ZercMoe"]
    # The fixed CANN source derives its kernel directory by inserting an
    # underscore before each uppercase letter. The public brand uses ZeRCMoE,
    # but its CANN OpType must follow this convention.
    derived_name = re.sub(r"(?<!^)(?=[A-Z])", "_", op_types[0]).lower()
    assert derived_name == "zerc_moe"
    assert re.search(r"\bOP_NAME\s+ZercMoe\b", cmake)
    assert re.search(r"\bOPTYPE\s+zerc_moe\b", cmake)
    assert re.search(r"\bvoid\s+zerc_moe\s*\(", kernel)
    assert not re.search(r"\bvoid\s+mega_moe\s*\(", kernel)
    graph_names = {
        node.value.value
        for node in ast.walk(graph)
        if isinstance(node, ast.keyword)
        and node.arg == "op_type"
        and isinstance(node.value, ast.Constant)
    }
    ir_names = {
        node.args[0].value
        for node in ast.walk(graph)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "IrDef"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert graph_names == ir_names == set(op_types)


def test_aclnn_binding_selects_isolated_entry_and_inner_pair():
    api = _read(OP, "op_api/aclnn_mega_moe.cpp")
    binding = _read(OPS, "csrc/mega_moe.cpp")
    names = _macro_args(binding, "ACLNN_CMD")
    assert names == ["aclnnZercMoe"]
    outer = names[0]
    inner = outer.replace("aclnn", "aclnnInner", 1)
    for name in (outer, inner):
        assert re.search(r"\b" + name + r"\s*\(", api)
        assert re.search(r"\b" + name + r"GetWorkspaceSize\s*\(", api)
    assert not re.search(r"\baclnn(?:Inner)?MegaMoe(?:GetWorkspaceSize)?\s*\(", api)


def test_torch_schema_jit_binding_and_converter_agree():
    wrapper = _read(OPS, "mega_moe.py")
    binding = _read(OPS, "csrc/mega_moe.cpp")
    graph = _read(OPS, "graph_convert/graph_convert_mega_moe.py")
    context = _read(OPS, "comm_context.py")
    context_cpp = _read(OPS, "csrc/comm_context.cpp")
    jit_names = re.findall(r'__init__\("(npu_\w+)"\)', wrapper)
    assert jit_names == ["npu_zerc_moe"]
    assert f'"{jit_names[0]}(Tensor context' in wrapper
    assert f"_op_module.{jit_names[0]}(" in wrapper
    assert f'm.def("{jit_names[0]}", &{jit_names[0]}' in binding
    assert f"torch.ops.zerc_moe.{jit_names[0]}.default" in graph
    assert 'self).__init__("zerc_moe_comm_context")' in context
    assert 'std::string(group) + "zerc_moe"' in context_cpp
    assert 'std::string(group) + "moe_dispatch_ffn_combine"' not in context_cpp


def test_comm_context_pybind_uses_a_unique_module_local_type():
    source = _read(OPS, "csrc/comm_context.cpp")
    registration = re.search(
        r'py::class_<([A-Za-z_]\w*)>\s*\(\s*m\s*,\s*"CommContextManager"\s*,\s*py::module_local\(\)\s*\)',
        source,
    )
    assert registration is not None
    class_name = registration.group(1)
    assert class_name == "ZercMoeCommContextManager"
    assert re.search(r"\bclass\s+" + class_name + r"\b", source)
    for method in ("create_context", "update_group", "destroy", "ccl_buffer_size"):
        assert f"&{class_name}::{method}" in source
