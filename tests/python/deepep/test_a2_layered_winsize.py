"""CPU-only consistency test for the A2 layered (hierarchy) HCCL window guard.

The layered dispatch/combine kernels hard-code the HCCL window layout
(800MB RDMA double buffer + 4MB IPC offset + 400MB notify region) and assert
a minimal window size (winSizeMin) in Init.  The assert is compiled out in
release builds, and on an undersized window the unsigned subtraction
    RANK_SIZE_ON_IPC = (totalSize_ - 800MB - 4MB - 400MB) / ...
wraps around into a huge offset, pushing shareAddrs out of the window and
crashing with "invalid GM address" (two weeks of forensics, 2026-11).

The guard now lives in three places that must stay in sync:
  * shared constants + host formulas:
      csrc/deepep/ops2/op_kernel/moe_distribute_a2_layered_winsize.h
  * kernel winSizeMin expressions + underflow clamps:
      csrc/deepep/ops2/op_kernel/moe_distribute_dispatch_v2_layered_custom.h
      csrc/deepep/ops2/op_kernel/moe_distribute_combine_v2_layered_custom.h
  * host tiling checks on the isLayered branch:
      csrc/deepep/ops2/op_host/moe_distribute_dispatch_v2_tiling.cc
      csrc/deepep/ops2/op_host/moe_distribute_combine_v2_tiling.cc

This test parses the actual expressions out of the C++ sources, evaluates
them for a sweep of realistic parameters, and checks:
  * the shared fixed overhead is exactly 1204MB and matches the kernels,
  * the host formulas are at least as strict as the kernel asserts for every
    supported dtype width,
  * the host tiling checks are wired on the layered branch of both operators,
  * the kernel underflow clamps are present,
  * the README layered formula names all three fixed terms.

It needs no NPU and runs anywhere:
    python tests/python/deepep/test_a2_layered_winsize.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS2 = REPO_ROOT / "csrc" / "deepep" / "ops2"
SHARED_H = OPS2 / "op_kernel" / "moe_distribute_a2_layered_winsize.h"
DISPATCH_KERNEL_H = OPS2 / "op_kernel" / "moe_distribute_dispatch_v2_layered_custom.h"
COMBINE_KERNEL_H = OPS2 / "op_kernel" / "moe_distribute_combine_v2_layered_custom.h"
DISPATCH_TILING_CC = OPS2 / "op_host" / "moe_distribute_dispatch_v2_tiling.cc"
COMBINE_TILING_CC = OPS2 / "op_host" / "moe_distribute_combine_v2_tiling.cc"
README_MD = REPO_ROOT / "python" / "deep_ep" / "doc" / "A2_DEEPEP.md"

MB = 1024 * 1024

_UPDATE_HINT = (
    "if the layered window layout changed on purpose, update "
    "moe_distribute_a2_layered_winsize.h, both kernel winSizeMin expressions, "
    "the README formula and this test together"
)


def _cxx_expr_to_py(expr: str) -> str:
    """Turn a simple C++ integer arithmetic expression into a Python one."""
    expr = re.sub(r"static_cast<[^>]+>", "", expr)  # static_cast<T>(x) -> (x)
    expr = expr.replace("MoeDistributeA2Layered::", "")
    expr = re.sub(r"sizeof\(uint32_t\)", "4", expr)
    expr = re.sub(r"sizeof\(XType\)", "X_TYPE_SIZE", expr)
    expr = re.sub(r"(\d+)\s*[uUlL]{1,3}\b", r"\1", expr)  # 800U/4UL -> 800/4
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def _eval(expr: str, env: dict) -> int:
    value = eval(_cxx_expr_to_py(expr), {"__builtins__": {}}, dict(env))  # noqa: S307
    assert isinstance(value, int), f"expression {expr!r} did not evaluate to an int"
    return value


def _parse_constexpr(src: str, name: str, env: dict, where: Path) -> int:
    m = re.search(
        r"constexpr\s+(?:static\s+)?[\w:]+\s+" + re.escape(name) + r"\s*=\s*([^;]+);",
        src,
    )
    assert m, f"constexpr {name} not found in {where.name} ({_UPDATE_HINT})"
    return _eval(m.group(1), env)


def _parse_statement_expr(src: str, pattern: str, what: str, path: Path) -> str:
    m = re.search(pattern, src, re.DOTALL)
    assert m, f"{what} not found in {path.name} ({_UPDATE_HINT})"
    return m.group(1)


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _load():
    shared_src = SHARED_H.read_text()
    consts = {}
    for name in (
        "RDMA_DATA_SIZE",
        "IPC_DATA_OFFSET",
        "NOTIFY_DATA_SIZE",
        "FIXED_WIN_OVERHEAD",
        "EXTRA_TOKEN_INFO_NUM",
        "TOKEN_INFO_UNIT_SIZE",
        "TOKEN_DTYPE_MAX_SIZE",
        "K_ALIGN_ENTRIES",
    ):
        consts[name] = _parse_constexpr(shared_src, name, consts, SHARED_H)
    return shared_src, consts


def _shared_formula(shared_src: str, func_name: str, consts: dict):
    body = _parse_statement_expr(
        shared_src,
        re.escape(func_name) + r"\([^)]*\)\s*\{\s*return\s+(.+?);\s*\}",
        f"{func_name} return expression",
        SHARED_H,
    )

    def calc(moe_expert_num: int, bs: int, h: int, k: int) -> int:
        env = dict(consts)
        env.update(
            moeExpertNum=moe_expert_num,
            bs=bs,
            h=h,
            k=k,
            AlignUp=_align_up,
        )
        return _eval(body, env)

    return calc


def _kernel_dispatch_min(kernel_src: str, consts: dict):
    expr = _parse_statement_expr(
        kernel_src,
        r"uint64_t winSizeMin =\s*(.+?);",
        "dispatch kernel winSizeMin expression",
        DISPATCH_KERNEL_H,
    )

    def calc(moe_expert_num: int, bs: int, h: int, k: int, x_type_size: int) -> int:
        env = {
            "moeExpertNum_": moe_expert_num,
            "axisBS_": bs,
            "axisH_": h,
            "alignK_": _align_up(k, consts["K_ALIGN_ENTRIES"]),
            "X_TYPE_SIZE": x_type_size,
            "EXTRA_TOKEN_INFO_NUM": consts["EXTRA_TOKEN_INFO_NUM"],
            "IPC_DATA_OFFSET": consts["IPC_DATA_OFFSET"],
            "RDMA_DATA_SIZE": consts["RDMA_DATA_SIZE"],
            "NOTIFY_DATA_SIZE": consts["NOTIFY_DATA_SIZE"],
        }
        return _eval(expr, env)

    return calc


def _kernel_combine_min(kernel_src: str, consts: dict):
    expr = _parse_statement_expr(
        kernel_src,
        r"uint64_t winSizeMin =\s*(.+?);",
        "combine kernel winSizeMin expression",
        COMBINE_KERNEL_H,
    )

    def calc(moe_expert_num: int, bs: int, h: int, k: int, x_type_size: int) -> int:
        env = {
            "moeExpertNum_": moe_expert_num,
            "axisBS_": bs,
            "axisK_": k,
            "axisHExpandXTypeSize_": h * x_type_size,
            "EXTRA_TOKEN_INFO_NUM": consts["EXTRA_TOKEN_INFO_NUM"],
            "IPC_DATA_OFFSET": consts["IPC_DATA_OFFSET"],
            "RDMA_DATA_SIZE": consts["RDMA_DATA_SIZE"],
            "NOTIFY_DATA_SIZE": consts["NOTIFY_DATA_SIZE"],
        }
        return _eval(expr, env)

    return calc


PARAM_SWEEP = [
    (moe_expert_num, bs, h, k)
    for moe_expert_num in (16, 64, 128, 160, 256)
    for bs in (1, 16, 64, 256, 512)
    for h in (1024, 2048, 4096, 7168)
    for k in (1, 4, 8, 13, 16)
]


def test_fixed_overhead_is_single_sourced():
    shared_src, consts = _load()
    assert consts["RDMA_DATA_SIZE"] == 800 * MB, _UPDATE_HINT
    assert consts["IPC_DATA_OFFSET"] == 4 * MB, _UPDATE_HINT
    assert consts["NOTIFY_DATA_SIZE"] == 400 * MB, _UPDATE_HINT
    assert consts["FIXED_WIN_OVERHEAD"] == 1204 * MB, _UPDATE_HINT

    # Both kernels must alias the shared constants, not redeclare literals.
    for path in (DISPATCH_KERNEL_H, COMBINE_KERNEL_H):
        src = path.read_text()
        for name in (
            "RDMA_DATA_SIZE",
            "IPC_DATA_OFFSET",
            "NOTIFY_DATA_SIZE",
            "EXTRA_TOKEN_INFO_NUM",
        ):
            m = re.search(
                r"constexpr\s+static\s+uint32_t\s+" + name + r"\s*=\s*([^;]+);", src
            )
            assert m, f"{name} not found in {path.name} ({_UPDATE_HINT})"
            assert f"MoeDistributeA2Layered::{name}" in m.group(1), (
                f"{name} in {path.name} no longer aliases the shared header "
                f"({_UPDATE_HINT})"
            )


def test_host_formula_covers_dispatch_kernel_assert():
    shared_src, consts = _load()
    host_calc = _shared_formula(shared_src, "CalcDispatchWinSizeMin", consts)
    kernel_calc = _kernel_dispatch_min(DISPATCH_KERNEL_H.read_text(), consts)
    for moe_expert_num, bs, h, k in PARAM_SWEEP:
        host_min = host_calc(moe_expert_num, bs, h, k)
        for x_type_size in (1, 2):
            kernel_min = kernel_calc(moe_expert_num, bs, h, k, x_type_size)
            assert host_min >= kernel_min, (
                f"host dispatch check weaker than kernel assert for "
                f"moeExpertNum={moe_expert_num} bs={bs} h={h} k={k} "
                f"sizeof(XType)={x_type_size}: host={host_min} kernel={kernel_min}"
            )
        # exact match at the dtype upper bound: no silent drift in either direction
        assert host_min == kernel_calc(
            moe_expert_num, bs, h, k, consts["TOKEN_DTYPE_MAX_SIZE"]
        ), f"host dispatch formula drifted from kernel winSizeMin ({_UPDATE_HINT})"


def test_host_formula_covers_combine_kernel_assert():
    shared_src, consts = _load()
    host_calc = _shared_formula(shared_src, "CalcCombineWinSizeMin", consts)
    kernel_calc = _kernel_combine_min(COMBINE_KERNEL_H.read_text(), consts)
    for moe_expert_num, bs, h, k in PARAM_SWEEP:
        host_min = host_calc(moe_expert_num, bs, h, k)
        for x_type_size in (1, 2):
            kernel_min = kernel_calc(moe_expert_num, bs, h, k, x_type_size)
            assert host_min >= kernel_min, (
                f"host combine check weaker than kernel assert for "
                f"moeExpertNum={moe_expert_num} bs={bs} h={h} k={k} "
                f"sizeof(ExpandXType)={x_type_size}: host={host_min} kernel={kernel_min}"
            )
        assert host_min == kernel_calc(
            moe_expert_num, bs, h, k, consts["TOKEN_DTYPE_MAX_SIZE"]
        ), f"host combine formula drifted from kernel winSizeMin ({_UPDATE_HINT})"


def test_host_tiling_checks_are_wired():
    for path, func in (
        (DISPATCH_TILING_CC, "CalcDispatchWinSizeMin"),
        (COMBINE_TILING_CC, "CalcCombineWinSizeMin"),
    ):
        src = path.read_text()
        blocks = re.findall(r"if \(isLayered\) \{(.*?)\n    \}", src, re.DOTALL)
        assert blocks, f"isLayered window check block missing in {path.name}"
        guard_blocks = [b for b in blocks if f"MoeDistributeA2Layered::{func}" in b]
        assert guard_blocks, f"{path.name} layered branch no longer calls {func}"
        block = guard_blocks[0]
        assert (
            "GetMaxWindowSize" in block
        ), f"{path.name} layered branch no longer reads the real window size"
        assert (
            "GRAPH_FAILED" in block
        ), f"{path.name} layered branch no longer rejects undersized windows"


def test_kernel_underflow_clamps_present():
    dispatch_src = DISPATCH_KERNEL_H.read_text()
    assert re.search(
        r"RANK_SIZE_ON_IPC =\s*\(totalSize_ > FIXED_WIN_OVERHEAD\)", dispatch_src
    ), f"dispatch RANK_SIZE_ON_IPC underflow clamp missing ({_UPDATE_HINT})"
    assert (
        "CheckWindowSize(winSizeMin" in dispatch_src
    ), f"dispatch release-mode CheckWindowSize guard missing ({_UPDATE_HINT})"

    combine_src = COMBINE_KERNEL_H.read_text()
    assert re.search(
        r"IPC_DATA_SIZE =\s*\(winContext_->winSize > FIXED_WIN_OVERHEAD\)", combine_src
    ), f"combine IPC_DATA_SIZE underflow clamp missing ({_UPDATE_HINT})"
    assert (
        "CheckWindowSize(winSizeMin" in combine_src
    ), f"combine release-mode CheckWindowSize guard missing ({_UPDATE_HINT})"


def test_incident_configuration_is_rejected():
    """Regression lock on the 2026-11 forensics scenario (HCCL_BUFFSIZE 400)."""
    shared_src, consts = _load()
    dispatch_calc = _shared_formula(shared_src, "CalcDispatchWinSizeMin", consts)
    combine_calc = _shared_formula(shared_src, "CalcCombineWinSizeMin", consts)

    # Any parameterisation needs more than the 1204MB fixed overhead, so a
    # 400MB window must always be rejected on the layered path.
    for calc in (dispatch_calc, combine_calc):
        for moe_expert_num, bs, h, k in PARAM_SWEEP:
            assert calc(moe_expert_num, bs, h, k) > 400 * MB

    # A representative dual-node config (~1232MB needed) passes with the
    # documented 1536MB setting and fails with 400MB.
    need = dispatch_calc(128, 16, 7168, 8)
    assert 1204 * MB < need <= 1536 * MB, f"unexpected need={need / MB:.1f}MB"
    assert need > 400 * MB


def test_readme_formula_mentions_all_fixed_terms():
    src = README_MD.read_text()
    hits = re.findall(r"4MB \+ 800MB \+ 400MB", src)
    assert len(hits) >= 2, (
        "README layered formula must name all three fixed terms "
        "(4MB IPC offset + 800MB RDMA + 400MB notify) in both the English and "
        "Chinese sections"
    )


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failed else 0)
