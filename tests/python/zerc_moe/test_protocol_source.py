"""Source-level checks for the window reuse handshake; these do not run kernels."""

import re
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "csrc/zerc_moe/source/mc2/mega_moe/op_kernel/mega_moe.h"
)


def _without_comments(source):
    # Preserve strings, so comment markers inside diagnostic messages are not
    # mistaken for C++ comments. No source is executed or compiled.
    pattern = re.compile(
        r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|//[^\n]*|/\*[\s\S]*?\*/'
    )
    return pattern.sub(lambda match: match.group(1) or " ", source)


def _function(source, qualified_name):
    start = source.index(qualified_name)
    opening = source.index("{", start)
    depth = 0
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : offset]
    raise AssertionError("Unbalanced source function body")


def test_combine_writers_join_before_reset_and_rank_handshake():
    code = _without_comments(SOURCE.read_text(encoding="utf-8"))
    process = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::Process()")
    combine = process.rindex("CombineTokensParallel<")
    reset = process.index("ResetTokenPerExpert();", combine)
    join = process.index("SyncAll<true>();", combine)
    assert combine < join < reset
    # A common AIV-only join follows the branch containing generic combine;
    # it must not be nested inside that branch alone.
    assert re.search(
        r"if\s+constexpr\s*\(g_coreType\s*==\s*AscendC::AIV\)\s*"
        r"\{\s*SyncAll<true>\(\);\s*\}",
        process[combine:reset],
    )


def test_histogram_reset_joins_before_cross_rank_publication():
    code = _without_comments(SOURCE.read_text(encoding="utf-8"))
    process = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::Process()")
    reset = process.index("ResetTokenPerExpert();")
    join = process.index("SyncAll<true>();", reset)
    handshake = process.index("CrossRankSyncInWorldSize();", reset)
    unpermute = process.index("UnpermuteCombineDedup();", handshake)
    assert reset < join < handshake < unpermute
    # The join is not guarded by a per-core predicate after reset.
    assert not re.search(r"\bif\s*(?:constexpr\s*)?\(", process[reset:join])


def test_rank_handshake_joins_before_advancing_ledger():
    code = _without_comments(SOURCE.read_text(encoding="utf-8"))
    handshake = _function(
        code, "MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()"
    )
    publish = handshake.index("WriteGmByPassDCache(syncRemoteAddr")
    wait = handshake.index("GmSignalWaitBarrier(syncCheck, count)", publish)
    join = handshake.index("SyncAll<true>();", wait)
    ledger = handshake.index("WriteGmByPassDCache(syncCount", join)
    assert publish < wait < join < ledger


def test_dense_tile_votes_follow_output_release():
    code = _without_comments(SOURCE.read_text(encoding="utf-8"))
    dense = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(")
    run = dense.index("GroupMatmul2MGroupStripe<")
    fallback_fence = dense.index("SyncFunc<AscendC::HardEvent::FIX_S>();", run)
    vote = dense.index("AtomicAdd(groupReadyAddr + GMM2_READY_OFFSET, int32_t(1))", run)
    assert run < fallback_fence < vote
    assert (
        "publishStripeReady ? groupReadyAddr + GMM2_STRIPE_READY_OFFSET : nullptr"
        in dense[run:fallback_fence]
    )
    assert re.search(r"if\s*\(!publishStripeReady\)", dense[run:fallback_fence])

    helper_path = SOURCE.with_name("mega_moe_catlass_impl.h")
    helper_code = _without_comments(helper_path.read_text(encoding="utf-8"))
    helper = _function(helper_code, "GroupMatmul2MGroupStripe(")
    stripe_release = helper[helper.index("if (stripeReadyAddr != nullptr)") :]
    # Paired, ragged-mask and ordinary stripe releases each fence before the
    # corresponding ready write. This checks source structure, not execution.
    paired = stripe_release.index("USE_PAIRED_STRIPE_RELEASE")
    paired_fence = stripe_release.index(
        "SyncFunc<AscendC::HardEvent::FIX_S>();", paired
    )
    paired_write = stripe_release.index("WriteGmByPassDCache(stripeReadyAddr", paired)
    ragged = stripe_release.index("USE_RAGGED_STRIPE_MASK", paired_write)
    ragged_fence = stripe_release.index(
        "SyncFunc<AscendC::HardEvent::FIX_S>();", ragged
    )
    ragged_write = stripe_release.index("WriteGmByPassDCache(maskAddr", ragged)
    ordinary_fence = stripe_release.index(
        "SyncFunc<AscendC::HardEvent::FIX_S>();", ragged_write
    )
    ordinary_write = stripe_release.index(
        "WriteGmByPassDCache(stripeReadyAddr + nTile", ordinary_fence
    )
    assert paired < paired_fence < paired_write < ragged < ragged_fence < ragged_write
    assert ragged_write < ordinary_fence < ordinary_write


def test_sparse_chunk_votes_are_width_counted_after_release():
    code = _without_comments(SOURCE.read_text(encoding="utf-8"))
    pipeline = _function(
        code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCoreWavePipeline("
    )
    sparse = pipeline[pipeline.index("if (!useDenseTileGrid)") :]
    run = sparse.index("GroupMatmul2MGroupStripe<")
    fence = sparse.index("SyncFunc<AscendC::HardEvent::FIX_S>();", run)
    vote = sparse.index("AtomicAdd(gmmAddrInfo.dispatchTileReady", fence)
    assert run < fence < vote
    assert "int32_t(chunkTiles)" in sparse[vote : vote + 300]


def _read_integer_expression(function_source, variable):
    """Read a uint32 C++ assignment used by the current scheduler."""
    match = re.search(
        r"\buint32_t\s+" + re.escape(variable) + r"\s*=\s*(.*?);",
        function_source,
        re.DOTALL,
    )
    assert match is not None, f"missing scheduler expression: {variable}"
    return re.sub(r"static_cast<(?:uint32_t|uint64_t)>", "", match.group(1)).strip()


def _arithmetic(expression, values):
    """Evaluate a limited nonnegative C++ integer expression without C++ execution.

    The test inputs fit in uint32/uint64, so casts do not truncate. Only the
    operators actually needed by the checked source are accepted.
    """
    import ast
    import operator

    expression = re.sub(r"\b(\d+)[uU]\b", r"\1", expression)
    if "?" in expression:
        condition, branches = expression.split("?", 1)
        yes, no = branches.split(":", 1)
        return _arithmetic(yes if _arithmetic(condition, values) else no, values)
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Mod: operator.mod,
    }
    comparisons = {ast.GtE: operator.ge}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in comparisons
        ):
            return comparisons[type(node.ops[0])](
                visit(node.left), visit(node.comparators[0])
            )
        raise AssertionError(f"Unsupported scheduler expression: {ast.dump(node)}")

    return visit(ast.parse(expression.strip(), mode="eval"))


def test_dense_tile_scheduler_assigns_every_tile_once():
    """Check unique ownership using the arithmetic expressions in the kernel.

    This model covers integer scheduling only, not NPU memory ordering or
    execution. Requiring the source loop and vote ties this property to the
    current implementation without requiring a device build.
    """
    from collections import Counter
    from itertools import product

    code = _without_comments(SOURCE.read_text(encoding="utf-8"))
    dense = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(")
    phase_expr = _read_integer_expression(dense, "groupPhase")
    first_expr = _read_integer_expression(dense, "firstTile")
    loop = re.search(
        r"for\s*\(\s*uint32_t\s+nTile\s*=\s*firstTile\s*;\s*"
        r"nTile\s*<\s*nTiles\s*;\s*nTile\s*\+=\s*(\w+)\s*\)",
        dense,
    )
    assert loop is not None, "dense source must enumerate each owned N tile"
    tile_loop = _function(dense[loop.start() :], "for (uint32_t nTile")
    assert "AtomicAdd(groupReadyAddr + GMM2_READY_OFFSET, int32_t(1))" in tile_loop
    step_name = loop.group(1)
    for cores, ntiles, group, phase in product(
        (1, 2, 3, 14, 28, 32),
        (1, 2, 7, 13, 28, 32, 33, 65),
        (0, 1, 17, 1024),
        (0, 1, 19, 55),
    ):
        values = {
            "groupOrdinal": group,
            "nTiles": ntiles,
            "gmm2StartBlockIdx": phase,
            "blockNum_": cores,
        }
        values["groupPhase"] = _arithmetic(phase_expr, values)
        owned_tiles = []
        for block in range(cores):
            values["blockIdx_"] = block
            first = _arithmetic(first_expr, values)
            assert first >= 0
            owned_tiles.extend(range(first, ntiles, values[step_name]))
        # Each tile must receive one owner and one ready vote, including tails
        # and cases where the number of tiles is not a multiple of the cores.
        assert Counter(owned_tiles) == Counter(range(ntiles))
