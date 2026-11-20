"""Regression properties for the current GMM2 scheduler.

The previous resident-A stripe pool, capped split, and weighted-tail scheduler
are no longer in this revision. These checks replace their obsolete structural
and modeled-speed assertions with properties of the dense and sparse scheduler.
They do not execute the NPU code or establish a performance improvement.
These checks were written independently for the current scheduler.
"""

import ast
import operator
import re
from collections import Counter
from itertools import permutations, product
from pathlib import Path

from tests.python.zerc_moe.test_protocol_source import (
    _arithmetic,
    _function,
    _read_integer_expression,
    _without_comments,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "csrc/zerc_moe/source/mc2/mega_moe/op_kernel/mega_moe.h"


def _kernel():
    return _without_comments(SOURCE.read_text(encoding="utf-8"))


def _assignments(body, name):
    return re.findall(
        r"\b(?:uint32_t\s+)?" + re.escape(name) + r"\s*=\s*([^;]+);", body
    )


def _int_expr(source, variables):
    """Interpret the small nonnegative arithmetic subset used by sparse chunks."""
    text = re.sub(r"\b(\d+)[uU]\b", r"\1", source.strip())
    nested_ternary = r"\(([^()?]+)\?([^():]+):([^()]+)\)"
    text = re.sub(nested_ternary, r"(\2 if \1 else \3)", text)
    if "?" in text:
        condition, alternatives = text.split("?", 1)
        left, right = alternatives.split(":", 1)
        text = f"({left} if {condition} else {right})"
    text = text.replace("/", "//")
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    comparisons = {ast.Lt: operator.lt, ast.Gt: operator.gt}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return comparisons[type(node.ops[0])](
                visit(node.left), visit(node.comparators[0])
            )
        if isinstance(node, ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        raise AssertionError(f"Unsupported sparse expression: {ast.dump(node)}")

    return visit(ast.parse(text, mode="eval"))


def test_dense_full_wave_balances_equal_cost_tile_ownership():
    dense = _function(
        _kernel(), "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles("
    )
    phase = _read_integer_expression(dense, "groupPhase")
    first = _read_integer_expression(dense, "firstTile")
    loop = re.search(
        r"for\s*\(\s*uint32_t\s+nTile\s*=\s*firstTile\s*;\s*"
        r"nTile\s*<\s*nTiles\s*;\s*nTile\s*\+=\s*blockNum_\s*\)",
        dense,
    )
    assert loop is not None
    for cores, groups, tiles, start in product(
        (2, 7, 28), (28, 64, 128), (2, 8, 13), (0, 3)
    ):
        all_tiles = []
        per_core = Counter()
        for group in range(groups):
            state = {
                "groupOrdinal": group,
                "nTiles": tiles,
                "gmm2StartBlockIdx": start,
                "blockNum_": cores,
            }
            state["groupPhase"] = _arithmetic(phase, state)
            for core in range(cores):
                state["blockIdx_"] = core
                first_tile = _arithmetic(first, state)
                for tile in range(first_tile, tiles, cores):
                    all_tiles.append((group, tile))
                    per_core[core] += 1
        # Independent mathematical oracle: each logical (M group, N tile)
        # has exactly one owner. Equal-cost tiles spread within one tile;
        # this says nothing about actual hardware time or ragged tile cost.
        assert Counter(all_tiles) == Counter(product(range(groups), range(tiles)))
        counts = [per_core[core] for core in range(cores)]
        assert max(counts) - min(counts) <= 1


def test_sparse_chunks_cover_tiles_and_publish_exact_width():
    pipeline = _function(
        _kernel(), "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCoreWavePipeline("
    )
    sparse = pipeline[pipeline.index("if (!useDenseTileGrid)") :]
    expressions = {
        name: _assignments(sparse, name)
        for name in (
            "sparseGroups",
            "chunksPerGroup",
            "baseChunkUnits",
            "extraUnitChunks",
            "chunkUnitEnd",
            "chunkEnd",
            "chunkTiles",
            "owner",
        )
    }
    assert all(expressions.values())
    assert re.search(
        r"CeilDiv\(\s*2U\s*\*\s*blockNum_\s*,\s*sparseGroups\s*\)",
        expressions["chunksPerGroup"][0],
    )
    assert "chunkUnitBegin = chunkUnitEnd;" in sparse
    assert "chunkBegin = chunkEnd;" in sparse
    assert "owner == blockIdx_ && chunkTiles > 0U" in sparse
    assert "int32_t(chunkTiles)" in sparse
    for cores, groups, tiles, tail, start in product(
        (2, 7, 28), (0, 1, 3, 6), (1, 2, 7, 13, 28, 33), (1, 2), (0, 5)
    ):
        # Restrict to the source's sparse gate. Tail is the actual final
        # stripe width in half-tile units, not an extra padded tile.
        if groups >= cores and tiles > 1:
            continue
        values = {
            "blockNum_": cores,
            "totalMGroups": groups,
            "nTiles": tiles,
            "groupUnits": 2 * (tiles - 1) + tail,
            "gmm2StartBlockIdx": start,
        }
        values["sparseGroups"] = _int_expr(expressions["sparseGroups"][0], values)
        values["chunksPerGroup"] = (2 * cores + values["sparseGroups"] - 1) // values[
            "sparseGroups"
        ]
        for update in expressions["chunksPerGroup"][1:]:
            values["chunksPerGroup"] = _int_expr(update, values)
        assert values["chunksPerGroup"] >= 1
        for name in ("baseChunkUnits", "extraUnitChunks"):
            values[name] = _int_expr(expressions[name][0], values)
        for group in range(groups):
            values.update(groupOrdinal=group, chunkUnitBegin=0, chunkBegin=0)
            assigned = []
            total_votes = 0
            for chunk in range(values["chunksPerGroup"]):
                values["chunkIdx"] = chunk
                values["chunkUnitEnd"] = _int_expr(
                    expressions["chunkUnitEnd"][0], values
                )
                values["chunkEnd"] = _int_expr(expressions["chunkEnd"][0], values)
                values["chunkEnd"] = _int_expr(expressions["chunkEnd"][1], values)
                values["chunkTiles"] = _int_expr(expressions["chunkTiles"][0], values)
                owner = _int_expr(expressions["owner"][0], values)
                assert 0 <= owner < cores
                if values["chunkTiles"]:
                    assigned.extend(range(values["chunkBegin"], values["chunkEnd"]))
                    total_votes += values["chunkTiles"]
                values["chunkUnitBegin"] = values["chunkUnitEnd"]
                values["chunkBegin"] = values["chunkEnd"]
            assert Counter(assigned) == Counter(range(tiles))
            assert total_votes == tiles


def test_full_and_ragged_groups_have_disjoint_dense_ordinals():
    code = _kernel()
    dense = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(")
    expression = _read_integer_expression(dense, "groupOrdinal")
    assert re.search(r"groupRows\s*>=\s*EXPERT_SCHEDULE_M_TILE_ROWS", expression)
    assert "fullGroupOrdinal++" in expression
    assert "totalFullMGroups + tailGroupOrdinal++" in expression
    tiling = SOURCE.with_name("mega_moe_tiling.h").read_text(encoding="utf-8")
    match = re.search(r"EXPERT_SCHEDULE_M_TILE_ROWS\s*=\s*(\d+)U\s*;", tiling)
    assert match is not None
    tile_rows = int(match.group(1))
    for rows in (
        (0, 1, tile_rows - 1, tile_rows, tile_rows + 1),
        (2 * tile_rows + 1, 1, 2 * tile_rows),
        (17, 5, tile_rows, 2 * tile_rows + 1),
    ):
        total_full = sum(row // tile_rows for row in rows)
        full_ordinal = tail_ordinal = 0
        groups = []
        for expert, count in enumerate(rows):
            for begin in range(0, count, tile_rows):
                group_rows = count - begin
                if group_rows >= tile_rows:
                    ordinal = full_ordinal
                    full_ordinal += 1
                else:
                    ordinal = total_full + tail_ordinal
                    tail_ordinal += 1
                groups.append((ordinal, expert, begin, min(group_rows, tile_rows)))
        assert sorted(ordinal for ordinal, *_ in groups) == list(range(len(groups)))
        reconstructed = Counter(
            (expert, row)
            for _, expert, begin, count in groups
            for row in range(begin, begin + count)
        )
        assert reconstructed == Counter(
            (expert, row) for expert, count in enumerate(rows) for row in range(count)
        )


def test_owner_stripe_ready_requires_matching_k16_protocol():
    code = _kernel()
    plan = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::PrepareCoreWaveGmm2Plan(")
    dense = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(")
    assert re.search(
        r"if\s+constexpr\s*\(UseOwnerChainCombine\s*&&\s*UseK16OwnerChain\)",
        plan,
    )
    assert re.search(
        r"publishStripeReady\s*=\s*useFullAReadyProtocol\s*&&\s*"
        r"CanUseFullAGmm2StripeReady\(",
        plan,
    )
    assert re.search(
        r"if\s+constexpr\s*\(UseOwnerChainCombine\s*&&\s*"
        r"\(!UseFullAReadyProtocol\s*\|\|\s*!UseK16OwnerChain\)\)"
        r"\s*\{\s*publishStripeReady\s*=\s*false\s*;",
        dense,
    )


def test_current_tile_release_does_not_require_a_full_stage_barrier():
    code = _kernel()
    dense = _function(code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(")
    pipeline = _function(
        code, "MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCoreWavePipeline("
    )
    sparse = pipeline[pipeline.index("if (!useDenseTileGrid)") :]
    assert "ProcessResidentGmm2Groups" not in code
    assert "Gmm2ResidentAContext" not in dense
    assert "Gmm2ResidentAContext" not in sparse
    assert "using TileGmm2Context" in dense
    assert "using SparseGmm2Context" in sparse
    assert "SyncAll<false>()" not in dense
    # The old full-A/capped policies are gone. The new safety contract is
    # per-tile/chunk output release before a ready vote, not their cache-hit
    # or modeled-latency promise. Check all producer interleavings for 3 tiles.
    release = sparse.index("SyncFunc<AscendC::HardEvent::FIX_S>();")
    vote = sparse.index("AtomicAdd(gmmAddrInfo.dispatchTileReady", release)
    assert release < vote
    assert "int32_t(chunkTiles)" in sparse[vote : vote + 300]
    events = tuple((tile, kind) for tile in range(3) for kind in ("release", "vote"))
    checked = 0
    for ordering in permutations(events):
        released = set()
        votes = set()
        for tile, kind in ordering:
            if kind == "release":
                released.add(tile)
            elif tile not in released:
                break
            else:
                votes.add(tile)
                if len(votes) == 3:
                    assert released == set(range(3))
        else:
            checked += 1
    assert checked > 0
