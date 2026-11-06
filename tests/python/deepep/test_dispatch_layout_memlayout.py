"""CPU-only consistency test for the DispatchLayoutA2 notify_send_data GM layout.

The notify_send_data buffer layout is computed independently in two places:
  * host side: csrc/deepep/deep_ep.cpp (notify_send_data_size allocation)
  * kernel side: csrc/deepep/ops2/op_kernel/dispatch_layout_a2.h (SetGlobalBuffer offsets)

There is no shared single source of truth, so the two sides can silently drift
apart.  That is exactly how the scratch-aliasing bug happened: the cross-core
prefix staging scratch (tempExpertGM_/tempServerGM_) aliased the head of the
per-expert token-index table (expertRankTokenIdxGM_), so a fast core writing
the output table raced with a slow core still reading its prefix scratch.

This test parses the actual expressions out of the C++ sources, evaluates them
for a sweep of realistic parameters, and checks them against a transcribed
region table:
  * all regions are consecutive and non-overlapping,
  * the scratch region starts strictly after the end of the token-index table
    (the de-alias regression tripwire),
  * the host allocation covers the scratch slots of every possible core count,
  * the constants duplicated between host and kernel stay in sync.

It needs no NPU and runs anywhere:  python tests/python/deepep/test_dispatch_layout_memlayout.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEEP_EP_CPP = REPO_ROOT / "csrc" / "deepep" / "deep_ep.cpp"
KERNEL_H = REPO_ROOT / "csrc" / "deepep" / "ops2" / "op_kernel" / "dispatch_layout_a2.h"
COMM_ARGS_H = REPO_ROOT / "csrc" / "deepep" / "ops2" / "op_kernel" / "comm_args.h"

_UPDATE_HINT = (
    "if the notify_send_data GM layout changed on purpose, update the region "
    "table in tests/python/deepep/test_dispatch_layout_memlayout.py to match"
)


def _cxx_expr_to_py(expr: str) -> str:
    """Turn a simple C++ integer arithmetic expression into a Python one."""
    expr = re.sub(r"static_cast<[^>]+>", "", expr)  # static_cast<T>(x) -> (x)
    expr = re.sub(r"(\d+)\s*[uUlL]{1,3}\b", r"\1", expr)  # 4096U -> 4096
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def _eval(expr: str, env: dict) -> int:
    value = eval(_cxx_expr_to_py(expr), {"__builtins__": {}}, dict(env))  # noqa: S307
    assert isinstance(value, int), f"expression {expr!r} did not evaluate to an int"
    return value


def _parse_constexpr(src: str, name: str, env: dict) -> int:
    m = re.search(
        r"constexpr\s+(?:static\s+)?[\w:]+\s+" + re.escape(name) + r"\s*=\s*([^;]+);",
        src,
    )
    assert m, f"constexpr {name} not found ({_UPDATE_HINT})"
    return _eval(m.group(1), env)


def _parse_statement_expr(src: str, pattern: str, what: str, path: Path) -> str:
    m = re.search(pattern, src, re.DOTALL)
    assert m, f"{what} not found in {path.name} ({_UPDATE_HINT})"
    return m.group(1)


def _load_sources():
    host_src = DEEP_EP_CPP.read_text()
    kernel_src = KERNEL_H.read_text()
    comm_src = COMM_ARGS_H.read_text()

    consts = {}
    consts["MAX_BATCH_SIZE"] = _parse_constexpr(host_src, "MAX_BATCH_SIZE", {})
    consts["EXPERT_DATA_SIZE"] = _parse_constexpr(host_src, "EXPERT_DATA_SIZE", consts)
    consts["LOCAL_RANK_SIZE"] = _parse_constexpr(host_src, "LOCAL_RANK_SIZE", {})
    consts["LAYOUT_SCRATCH_CORES"] = _parse_constexpr(
        host_src, "LAYOUT_SCRATCH_CORES", {}
    )
    consts["MAX_CORE_NUM"] = _parse_constexpr(comm_src, "MAX_CORE_NUM", {})
    consts["KERNEL_MAX_BATCH_SIZE"] = _parse_constexpr(kernel_src, "MAX_BATCH_SIZE", {})

    exprs = {
        "host_size": _parse_statement_expr(
            host_src,
            r"const int notify_send_data_size =\s*(.+?);",
            "notify_send_data_size expression",
            DEEP_EP_CPP,
        ),
        "item8_base": _parse_statement_expr(
            kernel_src,
            r"expertRankTokenIdxGM_\.SetGlobalBuffer\(\(__gm__ T \*\)notifySendData \+ (.+?)\);",
            "expertRankTokenIdxGM_ offset",
            KERNEL_H,
        ),
        "scratch_base": _parse_statement_expr(
            kernel_src,
            r"uint64_t scratchBase =\s*(.+?);",
            "scratchBase expression",
            KERNEL_H,
        ),
        "temp_expert_base": _parse_statement_expr(
            kernel_src,
            r"tempExpertGM_\.SetGlobalBuffer\(\(__gm__ T \*\)notifySendData \+ (.+?)\);",
            "tempExpertGM_ offset",
            KERNEL_H,
        ),
        "temp_server_base": _parse_statement_expr(
            kernel_src,
            r"tempServerGM_\.SetGlobalBuffer\(\(__gm__ T \*\)notifySendData \+ (.+?)\);",
            "tempServerGM_ offset",
            KERNEL_H,
        ),
    }
    return consts, exprs


def _spec_regions(E, S, K, aiv, mbs):
    """Transcription of the notify_send_data layout (ops2 dispatch_layout_a2.h).

    Returns an ordered list of (name, start, size) in int32 elements.
    """
    regions = []
    off = 0

    def add(name, size):
        nonlocal off
        regions.append((name, off, size))
        off += size

    add("1.num_tokens_per_expert", E)
    add("2.server_uniq_count", S)
    add("3.server_total_count", mbs * S)
    add("4.token_server_num", mbs)
    add("5.token_server_offset", mbs * S)
    add("6.send_token_idx", mbs * K)
    add("7.token_idx", mbs * K)
    add("8.expert_rank_token_idx", E * mbs)
    add("9.scratch_temp_expert", aiv * E)
    add("10.scratch_temp_server", aiv * S)
    return regions


def _param_sweep(consts):
    """Realistic A2 parameter combinations.

    num_ranks is a multiple of LOCAL_RANK_SIZE (8) and num_experts a multiple
    of num_ranks, so num_experts is always a multiple of 8; the kernel's
    32B-aligned scratch copies rely on this (an unaligned numExperts_ would
    spill into the neighbouring scratch slot).
    """
    local = consts["LOCAL_RANK_SIZE"]
    for num_ranks in (16, 32, 64, 128):
        for experts_per_rank in (1, 2, 4, 8):
            E = num_ranks * experts_per_rank
            if E > 1024:
                continue
            S = num_ranks // local
            for K in (4, 8, 16):
                yield E, S, K


def test_constants_in_sync():
    consts, _ = _load_sources()
    assert consts["LAYOUT_SCRATCH_CORES"] == consts["MAX_CORE_NUM"], (
        "LAYOUT_SCRATCH_CORES in deep_ep.cpp must equal Moe::MAX_CORE_NUM in "
        "comm_args.h: the host reserves one scratch slot per possible AIV core"
    )
    assert (
        consts["MAX_BATCH_SIZE"] == consts["KERNEL_MAX_BATCH_SIZE"]
    ), "MAX_BATCH_SIZE differs between deep_ep.cpp and dispatch_layout_a2.h"
    assert consts["EXPERT_DATA_SIZE"] == 1 + consts["MAX_BATCH_SIZE"]


def test_layout_matches_host_allocation():
    consts, exprs = _load_sources()
    mbs = consts["MAX_BATCH_SIZE"]
    max_cores = consts["MAX_CORE_NUM"]

    for E, S, K in _param_sweep(consts):
        host_env = {
            "num_experts": E,
            "server_num": S,
            "num_topk": K,
            "EXPERT_DATA_SIZE": consts["EXPERT_DATA_SIZE"],
            "MAX_BATCH_SIZE": mbs,
            "LAYOUT_SCRATCH_CORES": consts["LAYOUT_SCRATCH_CORES"],
        }
        host_size = _eval(exprs["host_size"], host_env)

        kernel_env = {
            "numExperts_": E,
            "serverNum_": S,
            "numTopk_": K,
            "MAX_BATCH_SIZE": mbs,
        }
        item8_base = _eval(exprs["item8_base"], kernel_env)
        scratch_base = _eval(exprs["scratch_base"], kernel_env)

        # blockDim depends on the SoC and on numTokens; the layout must hold
        # for every core count the tiling can produce.
        for aiv in (1, 2, 7, 40, max_cores):
            env = dict(kernel_env, scratchBase=scratch_base, aivNum_=aiv)
            temp_expert_base = _eval(exprs["temp_expert_base"], env)
            temp_server_base = _eval(exprs["temp_server_base"], env)

            spec = _spec_regions(E, S, K, aiv, mbs)
            spec_by_name = {name: (start, size) for name, start, size in spec}
            ctx = f"(E={E}, S={S}, K={K}, aiv={aiv})"

            # 1. De-alias regression tripwire: the scratch region must start
            #    at or after the END of the per-expert token-index table.
            #    (The old buggy code had temp_expert_base == item8_base.)
            item8_end = item8_base + E * mbs
            assert temp_expert_base >= item8_end, (
                f"scratch region aliases the expert token-index table {ctx}: "
                f"temp_expert_base={temp_expert_base} < item8_end={item8_end}"
            )
            assert temp_server_base >= item8_end, (
                f"scratch region aliases the expert token-index table {ctx}: "
                f"temp_server_base={temp_server_base} < item8_end={item8_end}"
            )

            # 2. The transcribed region table matches the parsed kernel offsets.
            assert item8_base == spec_by_name["8.expert_rank_token_idx"][0], ctx
            assert temp_expert_base == spec_by_name["9.scratch_temp_expert"][0], ctx
            assert temp_server_base == spec_by_name["10.scratch_temp_server"][0], ctx
            assert temp_expert_base == scratch_base, ctx

            # 3. Regions are consecutive and non-overlapping.
            for (name_a, start_a, size_a), (name_b, start_b, _) in zip(spec, spec[1:]):
                assert (
                    start_a + size_a == start_b
                ), f"regions {name_a} and {name_b} are not consecutive {ctx}"

            # 4. Every scratch slot of every core fits in the host allocation.
            scratch_end = temp_server_base + aiv * S
            assert scratch_end <= host_size, (
                f"scratch slots overrun notify_send_data {ctx}: "
                f"end={scratch_end} > host_size={host_size}"
            )

        # 5. The host formula reserves exactly the layout plus the full
        #    LAYOUT_SCRATCH_CORES scratch tail.
        expected = scratch_base + consts["LAYOUT_SCRATCH_CORES"] * (E + S)
        assert host_size == expected, (
            f"host notify_send_data_size does not match the kernel layout "
            f"(E={E}, S={S}, K={K}): host={host_size}, expected={expected}"
        )

        # num_experts is a multiple of 8 on A2 (num_ranks % 8 == 0 and
        # num_experts % num_ranks == 0), which the kernel's 32B-aligned
        # scratch DataCopyPad lengths rely on.
        assert E % 8 == 0, f"sweep produced an unaligned num_experts={E}"


def main():
    test_constants_in_sync()
    test_layout_matches_host_allocation()
    print("PASS: dispatch layout host/kernel memory layout is consistent")


if __name__ == "__main__":
    main()
