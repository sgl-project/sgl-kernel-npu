"""CPU source checks of clamp plumbing and vector paths.

These checks inspect source structure and do not compile or run device kernels.
"""

import ast
import re
from pathlib import Path

from tests.python.zerc_moe.test_protocol_source import _function, _without_comments

ROOT = Path(__file__).resolve().parents[3] / "csrc/zerc_moe/source"
OP = ROOT / "mc2/mega_moe"
EXT = ROOT / "torch_extension/npu_ops_transformer/ops"


def _read(path):
    return _without_comments(path.read_text(encoding="utf-8"))


def _flat(source):
    return re.sub(r"\s+", " ", source)


def _check_clamps_before_silu(body, gate, up, mask, neg, count):
    # Trace each vector lane from its own clamp block to its SiLU invocation.
    # This detects a missing lane, a lower bound on gate, or a missing up bound.
    body = _flat(body)
    starts = list(re.finditer(rf"AscendC::Reg::Muls\(\s*{neg}\s*,\s*{gate}\s*,", body))
    assert len(starts) == count
    previous = 0
    for start in starts:
        prefix = body[previous : start.start()]
        guard = prefix.rfind("if constexpr (HasClamp)")
        assert guard >= 0
        clamp = prefix[guard:]
        for call in (
            f"AscendC::Reg::Mins({gate}, {gate}, activationClamp, {mask});",
            f"AscendC::Reg::Maxs({up}, {up}, -activationClamp, {mask});",
            f"AscendC::Reg::Mins({up}, {up}, activationClamp, {mask});",
        ):
            assert call in clamp
        assert f"AscendC::Reg::Maxs({gate}" not in clamp
        assert clamp.index(f"AscendC::Reg::Maxs({up}") < clamp.index(
            f"AscendC::Reg::Mins({up}"
        )
        previous = start.end()


def test_clamp_precedes_silu_in_every_vector_path():
    bf16 = _read(OP / "op_kernel/block_epilogue_swiglu_bf16.h")
    bf16_swiglu = _function(bf16, "VfSwigluInPlace(")
    _check_clamps_before_silu(
        bf16_swiglu, "firstFp32", "secondFp32", "maskFp32", "negative", 2
    )

    mx = _read(OP / "op_kernel/block_epilogue_swiglu_mx_quant.h")
    generic = _function(
        mx,
        "BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VfSwiglu(",
    )
    # The generic function has one full-vector and one tail-vector loop.
    full = generic[: generic.index("for (uint16_t aa = 0; aa < dim1TailTimes; aa++)")]
    tail = generic[generic.index("for (uint16_t aa = 0; aa < dim1TailTimes; aa++)") :]
    _check_clamps_before_silu(full, "vregX1F", "vregX2F", "mask", "negReg", 1)
    _check_clamps_before_silu(tail, "vregX1F", "vregX2F", "mask1", "negReg", 1)
    fast = _function(
        mx,
        "BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VfSwigluAndMaxExp256(",
    )
    _check_clamps_before_silu(
        fast, "firstFp32", "secondFp32", "maskFp32", "negative", 4
    )


def test_zero_clamp_selects_the_unclamped_specializations():
    bf16 = _read(OP / "op_kernel/block_epilogue_swiglu_bf16.h")
    bf16_run = _flat(_function(bf16, "operator()("))
    assert re.search(
        r"if \(params_\.activationClamp > 0\.0f\) \{ VfSwigluInPlace<true>\(.*?params_\.activationClamp\); \} else \{ VfSwigluInPlace<false>\(.*?0\.0f\); \}",
        bf16_run,
    )
    mx = _read(OP / "op_kernel/block_epilogue_swiglu_mx_quant.h")
    mx_run = _flat(
        _function(
            mx,
            "BlockEpilogueSwigluMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(",
        )
    )
    for name in ("VfSwiglu", "VfSwigluAndMaxExp256"):
        assert re.search(
            rf"if \(params_\.activationClamp > 0\.0f\) \{{ {name}<true>\(.*?params_\.activationClamp\); \}} else \{{ {name}<false>\(.*?0\.0f\); \}}",
            mx_run,
        )
    assert mx_run.index("VfSwigluAndMaxExp256<true>") < mx_run.index(
        "ComputeScale(streamedMaxExpAddr"
    )
    assert mx_run.index("VfSwiglu<true>") < mx_run.index("ComputeMaxExp(gluResAddr")


def test_tiling_value_reaches_both_routed_and_shared_epilogues():
    kernel = _read(OP / "op_kernel/mega_moe.h")
    init = _flat(_function(kernel, "MegaMoe<TemplateMegaMoeTypeFunc>::Init(GM_ADDR"))
    for name, count in (
        ("sharedEpilogueOp_", 2),
        ("epilogueOp_", 1),
        ("streamingEpilogueOp_", 1),
    ):
        calls = re.findall(rf"(?<![\w]){name}\.Init\(\s*\{{([^}}]*)\}}\s*\)", init)
        assert len(calls) == count
        assert all(
            call.rstrip().endswith("tilingData->activationClamp") for call in calls
        )
    for file in ("block_epilogue_swiglu_bf16.h", "block_epilogue_swiglu_mx_quant.h"):
        source = _read(OP / "op_kernel" / file)
        assert "float activationClamp = 0.0f;" in source
        assert "params_ = params;" in source


def test_attr_index_and_aclnn_argument_order_are_consistent():
    opdef = _read(OP / "op_host/mega_moe_def.cpp")
    attrs = re.findall(r'this->Attr\("([^"]+)"\)', opdef)
    assert attrs[-2:] == ["shared_expert_num", "activation_clamp"]
    assert re.search(
        r'this->Attr\("activation_clamp"\)\.AttrType\(OPTIONAL\)\.Float\(0\.0f\)', opdef
    )
    config = _read(OP / "op_host/op_tiling/mega_moe.h")
    m = re.search(r"attrActivationClampIndex\s*=\s*(\d+)U", config)
    assert m and int(m.group(1)) == attrs.index("activation_clamp")
    tiling = _read(OP / "op_host/op_tiling/mega_moe_tiling.cpp")
    set_attr = _flat(_function(tiling, "SetAttrParams("))
    assert "attrs->GetAttrPointer<float>(config.attrActivationClampIndex)" in set_attr
    assert "tilingData->activationClamp = activationClamp;" in set_attr
    assert "!std::isfinite(activationClamp) || activationClamp < 0.0f" in set_attr
    assert "float activationClamp;" in _read(OP / "op_kernel/mega_moe_tiling.h")

    api = _flat(_read(OP / "op_api/aclnn_mega_moe.cpp"))
    assert (
        len(
            re.findall(
                r"int64_t globalBs, int64_t sharedExpertNum, double activationClamp, aclTensor \*yOut",
                api,
            )
        )
        == 2
    )
    assert (
        "commAlg, globalBs, sharedExpertNum, activationClamp, yOut, expertTokenNumsOut"
        in api
    )
    binding = _flat(_read(EXT / "csrc/mega_moe.cpp"))
    assert (
        "shared_expert_num, static_cast<double>(activationClamp), y, expert_token_nums"
        in binding
    )


def test_python_schema_and_graph_converter_both_forward_clamp():
    binding = (EXT / "mega_moe.py").read_text(encoding="utf-8")
    assert "float activation_clamp=0." in binding
    module = ast.parse(binding)
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    for name, target in (
        ("mega_moe", "_npu_mega_moe"),
        ("_npu_mega_moe", "npu_zerc_moe"),
    ):
        fn = functions[name]
        call = next(
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == target)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == target)
            )
        )
        assert (
            isinstance(call.args[-1], ast.Name)
            and call.args[-1].id == "activation_clamp"
        )
    graph = (EXT / "graph_convert/graph_convert_mega_moe.py").read_text(
        encoding="utf-8"
    )
    assert '"activation_clamp": attr.Float(activation_clamp)' in graph
    assert '.attr("activation_clamp", attr.Float(0.0))' in graph
    assert "activation_clamp=activation_clamp" in graph
