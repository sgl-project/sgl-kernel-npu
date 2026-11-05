"""CPU-only source and integer-model checks; no device synchronization is exercised."""

import ast
import itertools
import operator
import re
from collections import Counter
from pathlib import Path

KERNEL = (
    Path(__file__).resolve().parents[3] / "csrc/zerc_moe/source/mc2/mega_moe/op_kernel"
)


def _source(name):
    text = (KERNEL / name).read_text(encoding="utf-8")
    return re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", " ", text)


def _body(text, name):
    start = text.index("{", text.index(name))
    depth = 0
    for pos in range(start, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return text[start + 1 : pos]
    raise AssertionError(f"unbalanced body: {name}")


def _expression(body, name):
    match = re.search(r"\buint32_t\s+" + name + r"\s*=\s*(.*?);", body, re.S)
    assert match, name
    return match.group(1)


def _evaluate(expression, values):
    """Interpret the scheduler's small integer expression subset, without C++."""
    expression = re.sub(r"\b(\d+)[uU]\b", r"\1", expression)
    expression = re.sub(
        r"\((\w+)\s*<\s*(\w+)\s*\?\s*1\s*:\s*0\)",
        r"(1 if \1 < \2 else 0)",
        expression,
    )
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.floordiv,
        ast.Mod: operator.mod,
    }

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Lt)
        ):
            return visit(node.left) < visit(node.comparators[0])
        raise AssertionError(f"unsupported C++ expression: {ast.dump(node)}")

    return visit(ast.parse(expression.strip(), mode="eval"))


def test_core_wave_credit_handoff_drains_active_and_empty_cores():
    pipeline = _body(_source("mega_moe.h"), "::ProcessCoreWavePipeline(")
    dense = _body(_source("mega_moe.h"), "::ProcessGmm2MnTiles(")
    catlass = _source("mega_moe_catlass_impl.h")
    handoff = _body(catlass, "struct Gmm2OutputHandoff")
    stripe = _body(catlass, "GroupMatmul2MGroupStripe(")

    dispatch = pipeline.index(
        "DispatchExperts(gmmAddrInfo, 0U, expertPerRank_, rankBaseRow)"
    )
    fence = pipeline.index("PipeBarrier<PIPE_ALL>();", dispatch)
    vote = pipeline.index("CORE_WAVE_DISPATCH_READY_OFFSET", fence)
    notify = pipeline.index("NotifyCube(GMM2_UB_READY_FLAG);", vote)
    join = pipeline.index("SyncAll<true>();", notify)
    combine = pipeline.index("CombineTokensParallel<", join)
    assert dispatch < fence < vote < notify < join < combine
    assert pipeline.count("SyncAll<true>();") == 1
    assert "SyncAll<false>();" not in pipeline
    assert "gmm2Plan = PrepareCoreWaveGmm2Plan(rankBaseRow);" in pipeline
    assert pipeline.index(
        "GroupMatmulSwigluQuant<", pipeline.index("AscendC::AIC")
    ) < pipeline.index("gmm2Plan = PrepareCoreWaveGmm2Plan(rankBaseRow);")
    for code, typename in ((pipeline, "SparseGmm2Context"), (dense, "TileGmm2Context")):
        start = code.index(typename + " gmm2Context(")
        transferred = code.index("gmm1VecSetSyncCom_ != 0, true", start)
        clear = code.index("gmm1VecSetSyncCom_ = 0;", transferred)
        assert start < transferred < clear
        assert "WaitForGmm1ExpertReady(" in code
    assert "WaitForVector();" in handoff
    assert "WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);" in handoff
    assert handoff.index("pending_ = 0U;") < handoff.index("~Gmm2OutputHandoff()")
    assert "if (pending_ != 0U)" in handoff
    assert "Arm();" in handoff[handoff.index("~Gmm2OutputHandoff()") :]
    assert stripe.index("context.outputHandoff.Arm();") < stripe.index(
        "context.blockMmad("
    )


def test_ready_protocol_shares_budget_and_consumer_target():
    base = _source("mega_moe_base.h")
    catlass = _source("mega_moe_catlass_impl.h")
    combine = _source("mega_moe_combine_send.h")
    producer = _body(catlass, "GroupMatmul2(")
    consumer = _body(combine, "ResolveCombineRowRange(")
    group_ready = _body(combine, "AreCombineRowGroupsReady(")
    assert "GetGmm2ReadyWaveCount(" in producer
    assert "GetGmm2ReadyWaveCount(" in consumer
    assert "GetGmm2ReadyPrefixGroupCount(" in producer
    assert "if constexpr (ForceEveryMGroupReady)" in consumer
    assert "readyWaveCount = expertMGroups;" in consumer
    assert "GMM2_READY_OFFSET) != readyTarget" in re.sub(r"\s+", " ", group_ready)
    assert "secondWaveTiles >= minimumSecondWaveTiles" in base
    assert "finalMGroupRows * 4U >= EXPERT_SCHEDULE_M_TILE_ROWS" in base
    assert "materialMGroups >= 2U && materialMGroups < mGroups" in base
    assert "allowLongStreaming && materialMGroups >= 3U" in base


def test_sparse_ragged_scheduler_coverage_and_ready_votes():
    code = _source("mega_moe.h")
    plan = _body(code, "::PrepareCoreWaveGmm2Plan(")
    pipeline = _body(code, "::ProcessCoreWavePipeline(")
    sparse = _body(pipeline, "if (!useDenseTileGrid)")
    expressions = {
        name: _expression(sparse, name)
        for name in (
            "baseChunkUnits",
            "extraUnitChunks",
            "chunkUnitEnd",
            "chunkEnd",
            "chunkTiles",
            "owner",
        )
    }
    assert "2U * (nTiles - 1U) + tailStripeUnits" in plan
    assert "Ops::Base::CeilDiv(2U * blockNum_, sparseGroups)" in sparse
    assert (
        "chunksPerGroup = chunksPerGroup < nTiles ? chunksPerGroup : nTiles;" in sparse
    )
    assert "chunksPerGroup = chunksPerGroup > 0U ? chunksPerGroup : 1U;" in sparse
    assert "if (owner == blockIdx_ && chunkTiles > 0U)" in sparse
    assert "chunkEnd = chunkEnd < nTiles ? chunkEnd : nTiles;" in sparse
    assert "chunkUnitBegin = chunkUnitEnd;" in sparse
    assert "chunkBegin = chunkEnd;" in sparse
    run = sparse.index("GroupMatmul2MGroupStripe<")
    fence = sparse.index("SyncFunc<AscendC::HardEvent::FIX_S>();", run)
    vote = sparse.index("int32_t(chunkTiles)", fence)
    assert run < fence < vote
    assert "SyncAll<" not in sparse
    assert not re.search(r"\b(?:2049|3200|28|32)U?\b", sparse)

    # Original tail and 28/32-core shape coverage, plus small and large grids.
    for ntiles, tail, cores, group_choice in itertools.product(
        (1, 2, 8, 9, 13, 16, 20, 28, 33, 65), (1, 2), (1, 2, 28, 32), (1, 4, 16, -1)
    ):
        groups = cores - 1 if group_choice == -1 else group_choice
        if not 0 < groups < cores:
            continue
        chunks = max(1, min(ntiles, (2 * cores + groups - 1) // groups))
        units = 2 * (ntiles - 1) + tail
        values = {
            "groupUnits": units,
            "chunksPerGroup": chunks,
            "nTiles": ntiles,
            "blockNum_": cores,
            "gmm2StartBlockIdx": 3,
        }
        values["baseChunkUnits"] = _evaluate(expressions["baseChunkUnits"], values)
        values["extraUnitChunks"] = _evaluate(expressions["extraUnitChunks"], values)
        owners = Counter()
        for group in range(groups):
            values.update(groupOrdinal=group, chunkUnitBegin=0, chunkBegin=0)
            tiles = []
            ready_votes = 0
            for chunk in range(chunks):
                values["chunkIdx"] = chunk
                values["chunkUnitEnd"] = _evaluate(expressions["chunkUnitEnd"], values)
                end = min(_evaluate(expressions["chunkEnd"], values), ntiles)
                values["chunkEnd"] = end
                width = _evaluate(expressions["chunkTiles"], values)
                owner = _evaluate(expressions["owner"], values)
                assert 0 <= owner < cores and width >= 1
                assert width == end - values["chunkBegin"]
                if width:
                    owners[owner] += 1
                    tiles.extend(range(values["chunkBegin"], end))
                    ready_votes += width
                values.update(chunkUnitBegin=values["chunkUnitEnd"], chunkBegin=end)
            assert tiles == list(range(ntiles))
            assert ready_votes == ntiles
        assert sum(owners.values()) == groups * chunks
        loads = [owners[core] for core in range(cores)]
        assert max(loads) - min(loads) <= 1
        if groups * chunks >= cores:
            assert all(loads)


def test_reset_join_prevents_late_clear_model():
    process = _body(_source("mega_moe.h"), "::Process()")
    reset = process.index("ResetTokenPerExpert();")
    join = process.index("SyncAll<true>();", reset)
    publish = process.index("CrossRankSyncInWorldSize();", join)
    assert reset < join < publish

    # Abstract one late resetting worker and a fast worker that publishes the
    # rank handshake. Without the join a peer can begin the next epoch before
    # that reset, and its new histogram entry can then be lost.
    events = ("reset", "join", "publish", "peer_write")
    possible_without_join = [
        order
        for order in itertools.permutations(events)
        if order.index("publish") < order.index("peer_write")
    ]
    assert any(
        order.index("peer_write") < order.index("reset")
        for order in possible_without_join
    )
    possible_with_join = [
        order
        for order in possible_without_join
        if order.index("reset") < order.index("join") < order.index("publish")
    ]
    assert possible_with_join
    assert all(
        order.index("reset") < order.index("peer_write") for order in possible_with_join
    )


def test_ready_wave_examples_cover_single_hybrid_and_long_modes():
    # Independent policy model: this does not execute the C++ implementation.
    # The companion source test checks that producer and consumer share it.
    def classify(rows, columns, cores, scheduled=False):
        full, tail = divmod(rows, 256)
        groups = full + bool(tail)
        material = full + (tail >= 64)
        tiles = (columns + 255) // 256
        jobs = groups * tiles
        if not jobs or not cores or tiles > cores:
            return 0
        if jobs <= cores:
            return 1
        if material >= 2 and material < groups and material * tiles <= cores:
            return 2
        if (
            jobs <= 2 * cores
            and jobs - cores >= (cores + 1) // 2
            and (tail or 256) >= 64
        ):
            return 2
        return 3 if scheduled and material >= 3 else 0

    for cores in (28, 32):
        for columns in (3072, 3200):
            assert classify(513, columns, cores) == 2
            assert classify(512, columns, cores) == 1
        assert classify(768, 4096, cores) == 2
    for rows, cols, cores, scheduled, expected in (
        (433, 4096, 28, False, 0),
        (433, 4096, 32, False, 1),
        (948, 2048, 28, False, 0),
        (948, 2048, 28, True, 3),
        (513, 3200, 28, True, 2),
        (1179, 4096, 28, True, 3),
    ):
        assert classify(rows, cols, cores, scheduled) == expected
