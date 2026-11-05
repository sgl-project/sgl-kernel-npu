"""Additional CPU-only checks of GMM1/GMM2 ready contracts in the source.

These inspect source structure; they do not execute the NPU kernels or establish
hardware memory ordering.
"""

import re
from pathlib import Path

KERNEL = (
    Path(__file__).resolve().parents[3] / "csrc/zerc_moe/source/mc2/mega_moe/op_kernel"
)


def _source(name):
    source = (KERNEL / name).read_text(encoding="utf-8")
    return re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", " ", source)


def _body(source, name):
    opening = source.index("{", source.index(name))
    depth = 0
    for pos in range(opening, len(source)):
        depth += (source[pos] == "{") - (source[pos] == "}")
        if not depth:
            return source[opening + 1 : pos]
    raise AssertionError(f"Unbalanced source body: {name}")


def test_generic_gmm2_input_ready_and_exclusive_vote_paths():
    producer = _body(_source("mega_moe_catlass_impl.h"), "GroupMatmul2(")
    loop = _body(producer, "for (uint32_t loopIdx = startLoopIdx;")
    input_wait = producer.index(
        "ReadGmByPassDCache(flagValueAddr) < static_cast<int32_t>(targetLoops)"
    )
    compute = producer.index("context.blockMmad(", input_wait)
    loop_compute = loop.index("context.blockMmad(")
    multi_guard = loop.index("if (publishReadyPerTile)", loop_compute)
    multi_fence = loop.index("SyncFunc<AscendC::HardEvent::FIX_S>();", multi_guard)
    multi_vote = loop.index("AtomicAdd(readyAddr, int32_t(1));", multi_fence)
    single_guard = producer.index(
        "if (readyWaveCount == 1U && startLoopIdx < coreLoops)"
    )
    single_fence = producer.index(
        "SyncFunc<AscendC::HardEvent::FIX_S>();", single_guard
    )
    single_vote = producer.index("AtomicAdd(readyAddr, int32_t(1));", single_fence)
    assert input_wait < compute < single_guard < single_fence < single_vote
    assert loop_compute < multi_guard < multi_fence < multi_vote
    assert "publishReadyPerTile = readyWaveCount >= 2U" in producer
    assert producer.index("for (uint32_t loopIdx = startLoopIdx;") < single_guard
    assert "GMM2_READY_OFFSET" in loop[multi_guard:multi_vote]
    assert "GMM2_READY_OFFSET" in producer[single_guard:single_vote]


def test_generic_producer_consumer_agree_on_budget_and_fallback():
    kernel = _source("mega_moe.h")
    interleaved = _body(kernel, "::ProcessInterleavedGmmPipeline(")
    generic = _body(kernel, "::ProcessGenericGmm2(")
    combine = _source("mega_moe_combine_send.h")
    group_call = interleaved.index("GroupMatmul2<")
    fallback = interleaved.index("if (!useMGroupReady)", group_call)
    fallback_body = _body(interleaved[fallback:], "if (!useMGroupReady)")
    fence = fallback_body.index("SyncFunc<AscendC::HardEvent::FIX_S>();")
    vote = fallback_body.index("AtomicAdd(", fence)
    assert fence < vote
    assert "gmm2ReadyFlag_" in fallback_body[vote:]
    assert "fitsMGroupReadyBudget = expertMGroups <= mGroupsPerWave" in interleaved
    assert (
        "allowLongMGroupReady = expertScheduleEnabled_ && fitsMGroupReadyBudget"
        in interleaved
    )
    assert (
        "fitsMGroupReadyBudget, allowLongMGroupReady"
        in interleaved[group_call:fallback]
    )
    assert "expertMGroups <= mGroupsPerWave" in combine
    assert "expertScheduleEnabled && expertMGroups <= mGroupsPerWave" in combine
    assert "WaitForGmm2MGroupReady(readyAddr, readyTarget)" in combine
    assert "else if (!useMGroupReady)" in combine
    assert "gmm2ReadyFlagBasePtr" in combine
    generic_call = generic.index("GroupMatmul2<")
    generic_fence = generic.index(
        "SyncFunc<AscendC::HardEvent::FIX_S>();", generic_call
    )
    generic_vote = generic.index("AtomicAdd(", generic_fence)
    assert "gmm2ReadyFlag_" in generic[generic_vote:]
    assert "true, false, false," in generic[generic_call:generic_fence]


def test_gmm1_publication_precedes_expert_ready_poll():
    kernel = _source("mega_moe.h")
    epilogue = _body(_source("block_epilogue_swiglu_mx_quant.h"), "::operator()(")
    copy_data = epilogue.index("CopyOutputFromUb2Gm(")
    copy_scale = epilogue.index("CopyScaleFromUb2GmCompact(", copy_data)
    fence = epilogue.index("HardEvent::MTE3_S", copy_scale)
    publish = epilogue.index("AtomicAdd(groupFlagListGmAddr_, 1)", fence)
    assert copy_data < copy_scale < fence < publish
    assert "if constexpr (PublishReady_)" in epilogue[copy_scale:publish]
    wait = _body(kernel, "::WaitForGmm1ExpertReady(")
    assert "if (waited)" in wait
    assert "return;" in wait[wait.index("if (waited)") : wait.index("uint32_t outputN")]
    assert "Ops::Base::CeilDiv(rows, MegaMoeCatlassImpl::L1_TILE_M)" in wait
    assert "Ops::Base::CeilDiv(outputN, MegaMoeCatlassImpl::L1_TILE_N)" in wait
    assert "ReadGmByPassDCache(gmmAddrInfo.flagSwigluToGmm2) <" in wait
    assert "GmSpinBackoffStep(" in wait
    assert "waited = true;" in wait
    for function in ("::ProcessCoreWavePipeline(", "::ProcessGmm2MnTiles("):
        source = _body(kernel, function)
        assert "bool waitedForGmm1Ready = false;" in source
        assert source.index("WaitForGmm1ExpertReady(") < source.index(
            "GroupMatmul2MGroupStripe<"
        )


def test_ready_workspace_and_resets_cover_both_cachelines():
    workspace = _source("mega_moe_workspace_info.h")
    kernel = _source("mega_moe.h")
    assert "DISPATCH_TILE_READY_CACHELINES = 2U" in workspace
    assert "DISPATCH_READY_OFFSET = 0U" in workspace
    assert "GMM2_READY_OFFSET = static_cast<uint32_t>(INT_CACHELINE)" in workspace
    assert (
        "GMM2_READY_OFFSET + static_cast<uint32_t>(INT_CACHELINE) == DISPATCH_TILE_READY_STRIDE"
        in workspace
    )
    assert "GMM2_READY_N_TILE_COLUMNS = 256U" in workspace
    assert workspace.count("GM_ADDR ptrDispatchTileReady;") == 1
    assert "ptrGmm2MGroupReady" not in workspace
    full_reset = _body(kernel, "::ResetDispatchTileReadyFull()")
    assert "maxDispatchTilesPerExpert_ * DISPATCH_TILE_READY_STRIDE" in full_reset
    assert "Duplicate(initTensor, 0, curFlags)" in full_reset
    assert "DataCopyPad(dispatchTileReady_[offset], initTensor" in full_reset
    partial_reset = _body(kernel, "::ResetActiveDispatchTileReady()")
    assert "Duplicate(initTensor, 0, DISPATCH_TILE_READY_STRIDE)" in partial_reset
    assert (
        "DataCopy(dispatchTileReady_[readyOffset], initTensor, DISPATCH_TILE_READY_STRIDE)"
        in partial_reset
    )
    reset = _body(kernel, "::ResetData()")
    assert reset.index("DataCopy(flagSwigluToGmm2_") < reset.index(
        "DataCopy(chainLenReady_"
    )
    assert "DataCopy(gmm2ReadyFlag_" in reset
    assert "ResetDispatchTileReadyFull();" in reset


def test_cross_rank_wait_uses_barrier_predicate_and_backoff():
    kernel = _source("mega_moe.h")
    wait = _body(kernel, "GmSignalWaitBarrier(")
    assert "IsCrossRankBarrierReached(" in wait
    assert "GmSpinBackoffStep(" in wait
    sync = _body(kernel, "::CrossRankSyncInWorldSize()")
    assert "if constexpr (!UseBf16NoQuant)" not in sync
    assert "handshakeOwners" not in sync
    assert sync.index("GmSignalWaitBarrier(syncCheck, count)") < sync.index(
        "SyncAll<true>();"
    )
    assert sync.index("SyncAll<true>();") < sync.index("WriteGmByPassDCache(syncCount")


def test_generic_ready_helper_arguments_and_tile_count_agree():
    kernel = _source("mega_moe.h")
    producer_source = _source("mega_moe_catlass_impl.h")
    producer = _body(producer_source, "GroupMatmul2(")
    consumer_source = _source("mega_moe_combine_send.h")
    planner = _body(consumer_source, "ResolveCombineRowRange(")
    consumer = _body(consumer_source, "CombineTokensParallel(")
    normalize = lambda text: re.sub(r"\s+", " ", text)
    producer = normalize(producer)
    planner = normalize(planner)
    consumer = normalize(consumer)
    interleaved = normalize(_body(kernel, "::ProcessInterleavedGmmPipeline("))
    init = normalize(_body(kernel, "::Init("))
    assert "k_ = tilingData->k;" in init
    assert "Get<K_VALUE>(problemShape_) = k_;" in init
    assert "Gmm2Context gmm2Context(Get<K_VALUE>(problemShape_)" in interleaved
    assert ": n(nValue)" in normalize(producer_source)
    assert "uint32_t m = Get<M_VALUE>(problemShape);" in producer
    group_params = _body(kernel, "::SetGroupParams(")
    assert "GetExpertRange(expertIdx, rankBaseRow, localStart, rows)" in group_params
    assert "Get<M_VALUE>(problemShape_) = rows;" in group_params
    producer_range = normalize(_body(kernel, "::GetExpertRange("))
    consumer_range = normalize(_body(consumer_source, "ClampCombineExpertRows("))
    assert "rows = static_cast<int32_t>(remainingRows);" in producer_range
    assert "rowsThisExpert = static_cast<int32_t>(remainingRows);" in consumer_range
    assert "uint32_t n = context.n;" in producer
    assert "uint32_t blockNum = context.blockNum;" in producer
    assert "blockNum(GetBlockNum())" in normalize(producer_source)
    assert "int32_t n = static_cast<int32_t>(params.tilingData->k);" in consumer
    assert "uint32_t blockNum = GetBlockNum();" in consumer
    assert (
        "uint32_t expertMGroups = GetGmm2MGroupCount(static_cast<uint32_t>(rowsThisExpert));"
        in planner
    )
    assert "if (expertMGroups <= mGroupsPerWave)" in planner
    assert "expertScheduleEnabled && expertMGroups <= mGroupsPerWave" in planner
    assert (
        "allowMGroupReady ? GetGmm2ReadyWaveCount(m, n, blockNum, allowLongMGroupReady) : 0U"
        in producer
    )
    assert (
        "GetGmm2ReadyWaveCount(static_cast<uint32_t>(rowsThisExpert), "
        "static_cast<uint32_t>(n), blockNum, allowLongStreaming)"
    ) in planner
    assert "useMGroupReady = readyWaveCount != 0U;" in planner
    assert "return readyWaveCount != 0U;" in producer
    assert "if constexpr (ForceEveryMGroupReady)" in planner
    assert "readyWaveCount = expertMGroups;" in planner
    assert "uint32_t nTiles = GetGmm2NTileCount(static_cast<uint32_t>(n));" in consumer
    assert "int32_t readyTarget = static_cast<int32_t>(nTiles);" in consumer
    assert "WaitForGmm2MGroupReady(readyAddr, readyTarget);" in consumer
    wait = _body(consumer_source, "WaitForGmm2MGroupReady(")
    assert "ReadGmByPassDCache(readyAddr) != atomicReadyTarget" in wait
    base = _body(_source("mega_moe_base.h"), "GetGmm2NTileCount(")
    assert "CeilDiv(columns, GMM2_READY_N_TILE_COLUMNS)" in base
    assert "L1_TILE_N == GMM2_READY_N_TILE_COLUMNS" in producer_source
