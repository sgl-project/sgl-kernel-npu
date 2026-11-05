"""CPU models and source checks for ZeRCMoE scheduling and plan protocol.

Adapted from tests/megamoe/test_megamoe_accuracy_cpu.py in
https://gitcode.com/hanggongmao1/ops-transformer, mega_moe branch,
revision d4c5f65d93591f26895c126fc590f0d622440819.
The source repository's CANN Open Software License Agreement Version 2.0 is
included with the operator at csrc/zerc_moe/LICENSE.
These checks do not execute the accelerator implementation.
"""

import random
import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[3] / "csrc/zerc_moe/source"


def test_generic_expert_schedule_is_shared_by_gmm_stages_and_combine() -> None:
    root = ROOT
    kernel_source = (root / "mc2/mega_moe/op_kernel/mega_moe.h").read_text(
        encoding="utf-8"
    )
    combine_source = (
        root / "mc2/mega_moe/op_kernel/mega_moe_combine_send.h"
    ).read_text(encoding="utf-8")

    process_start = kernel_source.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::Process()"
    )
    process_source = kernel_source[process_start:]
    prepare_start = kernel_source.index(
        "int32_t MegaMoe<TemplateMegaMoeTypeFunc>::PrepareExpertPipeline()"
    )
    prepare_end = kernel_source.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCoreWavePipeline(",
        prepare_start,
    )
    prepare_source = kernel_source[prepare_start:prepare_end]
    assert "if constexpr (UseCoreWave)" in prepare_source
    assert "expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS" in prepare_source
    for shape_pin in (
        "m_ == 2597U",
        "worldSize_ == 4U",
        "expertPerRank_ == 48U",
        "topK_ == 8U",
        "k_ == 4096U",
        "Get<N_VALUE>(problemShape_) == 3072U",
        "maxOutputSize_ == 83104ULL",
    ):
        assert shape_pin not in prepare_source

    gmm2_start = kernel_source.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGenericGmm2("
    )
    gmm2_end = kernel_source.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareSharedExpertInput(", gmm2_start
    )
    gmm2_source = kernel_source[gmm2_start:gmm2_end]
    assert "ProcessGenericGmm2<true>(rankBaseRow)" in process_source
    assert "ProcessGenericGmm2<false>(rankBaseRow)" in process_source
    for stage in (process_source, gmm2_source):
        assert (
            stage.count(
                "for (uint32_t scheduleIdx = 0; scheduleIdx < expertPerRank_; scheduleIdx++)"
            )
            == 1
        )
        assert stage.count("GetScheduledExpert(scheduleIdx)") == 1
        assert stage.count("expertScheduleEnabled_ ?") == 1
        assert stage.count("SetGroupParams(expertIdx, rankBaseRow)") == 1
        assert stage.count("UpdateIdentityGroupParams(expertIdx)") == 1
    assert "GroupMatmulSwigluQuant<" in process_source
    assert "GroupMatmul2<" in gmm2_source
    assert "gmm1VecSetSyncCom_, timer, expertIdx" in process_source
    assert "startBlockIdx_, expertIdx" in gmm2_source
    assert "params_.workspaceInfo.ptrExpertOrder" in process_source
    assert (
        "expertScheduleEnabled_, 0U, expertPerRank_, 0U, " "blockAivNum_"
    ) in " ".join(process_source.split())
    for removed_balanced_mode in (
        "balancedGmmPhaseEnabled_",
        "balancedExpertEndpointsEnabled_",
        "balancedBaseRows_",
        "balancedExtraExperts_",
        "gmm1PhaseStride_",
        "gmm2PhaseStride_",
    ):
        assert removed_balanced_mode not in kernel_source

    assert (
        "for (uint32_t scheduleIdx = scheduleBegin; scheduleIdx < scheduleEnd; ++scheduleIdx)"
        in combine_source
    )
    assert "expertOrderPtr + scheduleIdx" in combine_source
    assert "useProfiledE48Warmup" not in combine_source
    assert "if (expertScheduleEnabled)" in combine_source
    assert "globalExpertIdx == 0 ? 0" in combine_source
    assert "if (!expertScheduleEnabled && scheduleBegin > 0U)" in combine_source
    assert "useBalancedEndpoints" not in combine_source
    assert "balancedOwnerRows" not in combine_source
    assert "balancedBaseRows" not in combine_source
    assert "gmm2ReadyFlagBasePtr +" in combine_source
    assert "static_cast<int32_t>(e) * INT_CACHELINE" in combine_source

    # Runtime distribution, rather than a named catalog shape, decides whether
    # the schedule is enabled. Uniform cases remain identity after the scan.
    assert "BuildExpertOrder();" not in prepare_source
    assert "LoadExpertScheduleMode();" in prepare_source
    assert "bool useHotFirstSchedule" in kernel_source
    assert "maxTiles - minTiles >= EXPERT_SCHEDULE_MIN_TILE_SPREAD" in kernel_source
    assert "allowLongMGroupReady = expertScheduleEnabled_" in kernel_source


def test_stable_quotient_and_weighted_group_partition_models_and_source() -> None:
    def logical_core(physical_core: int, phase: int, core_count: int) -> int:
        return (physical_core + phase) % core_count

    def quotient_ranges(rows: int, core_count: int, phase: int) -> List[List[int]]:
        base, remainder = divmod(rows, core_count)
        assigned: List[List[int]] = []
        for physical_core in range(core_count):
            logical = logical_core(physical_core, phase, core_count)
            begin = logical * base + min(logical, remainder)
            count = base + int(logical < remainder)
            assigned.append(list(range(begin, begin + count)))
        return assigned

    def schedule_phase(schedule_idx: int, core_count: int, *, scheduled: bool) -> int:
        del schedule_idx, core_count, scheduled
        return 0

    for core_count in (1, 2, 3, 6, 8, 10, 14, 28, 56, 64):
        for rows in (1, 55, 255, 256, 257, 432, 433, 512, 513, 697, 1024, 1179):
            for schedule_idx in range(min(core_count, 16)):
                phase = schedule_phase(schedule_idx, core_count, scheduled=True)
                assigned = quotient_ranges(rows, core_count, phase)
                flat = [row for part in assigned for row in part]
                assert sorted(flat) == list(range(rows))
                assert len(flat) == len(set(flat))
                loads = [len(part) for part in assigned]
                assert max(loads) - min(loads) <= 1
                assert all(
                    not part or part == list(range(part[0], part[-1] + 1))
                    for part in assigned
                )
                assert sorted(
                    logical_core(core, phase, core_count) for core in range(core_count)
                ) == list(range(core_count))

        # Both identity and hot/cold traversal retain stable physical owners;
        # a token chain must not bounce across AIV cache domains.
        identity_owners = {
            logical_core(
                7 % core_count,
                schedule_phase(schedule_idx, core_count, scheduled=False),
                core_count,
            )
            for schedule_idx in range(max(core_count, 1))
        }
        assert len(identity_owners) == 1
        scheduled_owners = {
            logical_core(
                7 % core_count,
                schedule_phase(schedule_idx, core_count, scheduled=True),
                core_count,
            )
            for schedule_idx in range(max(core_count, 1))
        }
        assert len(scheduled_owners) == 1

    def use_group_partition(
        rows: int,
        core_count: int,
        batch: int,
        *,
        ready: bool,
        multi_wave: bool = False,
        long_streaming: bool = False,
    ) -> bool:
        groups = [min(256, rows - start) for start in range(0, rows, 256)]
        material_groups = rows // 256 + int(rows % 256 >= 64)
        if len(groups) > core_count:
            return False
        if material_groups != len(groups):
            return False
        whole_max_rows = (rows + core_count - 1) // core_count
        whole_batches = (whole_max_rows + batch - 1) // batch
        extra_workers = core_count - len(groups)
        prefix_rows = 0
        grouped_batches = 0
        grouped_max_rows = 0
        for group_idx, group_rows in enumerate(groups):
            worker_begin = group_idx + prefix_rows * extra_workers // rows
            prefix_rows += group_rows
            worker_end = group_idx + 1 + prefix_rows * extra_workers // rows
            group_workers = worker_end - worker_begin
            max_group_rows = (group_rows + group_workers - 1) // group_workers
            grouped_max_rows = max(grouped_max_rows, max_group_rows)
            grouped_batches = max(
                grouped_batches,
                (max_group_rows + batch - 1) // batch,
            )
        return (
            ready
            and multi_wave
            and len(groups) > 1
            and grouped_max_rows <= whole_max_rows + int(long_streaming)
            and grouped_batches <= whole_batches
        )

    def weighted_group_workers(rows: int, core_count: int) -> List[int]:
        groups = [min(256, rows - start) for start in range(0, rows, 256)]
        assert len(groups) <= core_count
        extra_workers = core_count - len(groups)
        prefix_rows = 0
        workers: List[int] = []
        for group_idx, group_rows in enumerate(groups):
            worker_begin = group_idx + prefix_rows * extra_workers // rows
            prefix_rows += group_rows
            worker_end = group_idx + 1 + prefix_rows * extra_workers // rows
            workers.append(worker_end - worker_begin)
        assert sum(workers) == core_count
        assert all(worker_count >= 1 for worker_count in workers)
        return workers

    # Material multi-wave experts gain concurrent ready-group ownership
    # without adding a slow-core DMA batch; long-ready experts may spend one
    # extra row to expose a materially earlier group. A 513-row expert uses
    # hybrid prefix readiness, but the one-row tail is not a material group,
    # so production must not select its otherwise feasible [27, 27, 2]
    # independent-group worker partition.
    assert not use_group_partition(348, 56, 4, ready=True)
    assert not use_group_partition(512, 56, 4, ready=True)
    assert use_group_partition(697, 56, 8, ready=True, multi_wave=True)
    assert use_group_partition(1024, 56, 8, ready=True, multi_wave=True)
    assert not use_group_partition(513, 56, 8, ready=True, multi_wave=True)
    assert use_group_partition(
        1176, 56, 8, ready=True, multi_wave=True, long_streaming=True
    )
    assert use_group_partition(
        1179, 56, 8, ready=True, multi_wave=True, long_streaming=True
    )
    assert not use_group_partition(432, 56, 4, ready=False)

    def group_partition_ranges(rows: int, core_count: int) -> List[List[int]]:
        workers = weighted_group_workers(rows, core_count)
        assigned: List[List[int]] = []
        for group_idx, count in enumerate(workers):
            begin = group_idx * 256
            group_rows = min(256, rows - begin)
            base, extra = divmod(group_rows, count)
            for worker_idx in range(count):
                start = begin + worker_idx * base + min(worker_idx, extra)
                length = base + int(worker_idx < extra)
                assigned.append(list(range(start, start + length)))
        return assigned

    # The arithmetic helper remains total for ragged inputs even when the
    # runtime readiness gate rejects that partition.
    assert weighted_group_workers(348, 56) == [40, 16]
    assert weighted_group_workers(349, 56) == [40, 16]
    assert weighted_group_workers(512, 56) == [28, 28]
    assert weighted_group_workers(513, 56) == [27, 27, 2]
    assert weighted_group_workers(697, 56) == [20, 20, 16]
    assert weighted_group_workers(1024, 56) == [14, 14, 14, 14]
    for core_count in (1, 2, 3, 7, 14, 28, 32, 56, 64):
        for rows in (
            1,
            55,
            56,
            92,
            93,
            255,
            256,
            257,
            348,
            349,
            512,
            513,
            697,
            1024,
            1179,
            2048,
            4096,
        ):
            group_count = (rows + 255) // 256
            if group_count > core_count:
                continue
            workers = weighted_group_workers(rows, core_count)
            assert len(workers) == group_count
            assert sum(workers) == core_count
            assigned = group_partition_ranges(rows, core_count)
            assert len(assigned) == core_count
            assert sorted(row for part in assigned for row in part) == list(range(rows))
            assert all(
                not part or part == list(range(part[0], part[-1] + 1))
                for part in assigned
            )
    for rows in (348, 349, 512, 697, 1024, 1179):
        groups = [min(256, rows - start) for start in range(0, rows, 256)]
        worker_counts = weighted_group_workers(rows, 56)
        grouped_max_rows = max(
            (group_rows + worker_count - 1) // worker_count
            for group_rows, worker_count in zip(groups, worker_counts)
        )
        assert grouped_max_rows <= (rows + 56 - 1) // 56 + 1

    root = ROOT
    combine = (root / "mc2/mega_moe/op_kernel/mega_moe_combine_send.h").read_text(
        encoding="utf-8"
    )
    kernel = (root / "mc2/mega_moe/op_kernel/mega_moe.h").read_text(encoding="utf-8")
    combine_start = combine.index("void CombineTokensParallel(")
    combine_end = combine.index(
        "// Legacy readiness-sweep token-owner experiment", combine_start
    )
    production_combine = combine[combine_start:combine_end]
    ready_wait_start = combine.index("void WaitForGmm2MGroupReady(")
    ready_wait_end = combine.index(
        "template <bool PublishAiv1UbOwnership>", ready_wait_start
    )
    ready_wait_source = combine[ready_wait_start:ready_wait_end]
    assert "nTiles" not in ready_wait_source
    assert "readyAddr + nTile" not in ready_wait_source
    assert "ReadGmByPassDCache(readyAddr) != atomicReadyTarget" in ready_wait_source
    assert production_combine.count("WaitForGmm2MGroupReady(") == 1
    assert "WaitForGmm2MGroupReadyThroughGate(" not in production_combine
    assert "consumerGate" not in production_combine
    assert "rotationStride" not in production_combine
    assert "coreRotation" not in production_combine
    expert_loop = production_combine.index("for (uint32_t scheduleIdx = scheduleBegin;")
    for invariant in (
        "__gm__ int32_t *dispatchTileReadyBasePtr",
        "uint32_t nTiles = GetGmm2NTileCount",
        "uint64_t maxTilesPerExpert = Ops::Base::CeilDiv",
        "uint64_t dispatchTileReadyExpertStride",
    ):
        assert production_combine.index(invariant) < expert_loop
        assert production_combine.count(invariant) == 1
    range_start = combine.index("CombineRowRange ResolveCombineRowRange(")
    range_end = combine.index("bool AreCombineRowGroupsReady(", range_start)
    range_source = combine[range_start:range_end]
    assert "GetCombineLogicalCore(" not in production_combine
    assert "uint32_t logicalCore = aivCoreIdx;" in range_source
    for invariant in (
        "bool partitionByReadyGroup",
        "bool canAssignOneGroupPerCore",
        "GetMaterialMGroupCount(",
        "readyWaveCount >= 2U",
        "uint32_t groupWorkerBegin",
        "groupedMaxRowsPerCore <= allowedGroupedRowsPerCore",
        "groupedRangeBatches <= wholeRangeBatches;",
        "uint32_t baseRows = static_cast<uint32_t>(rowsThisExpert) /",
    ):
        assert invariant in range_source
    assert (
        "ResolveCombineRowRange<EnableMGroupReady, ForceEveryMGroupReady>("
        in production_combine
    )
    assert (
        "ClampCombineExpertRows(expertTokenStart, maxOutputSize, rowsThisExpert)"
        in production_combine
    )
    for field in ("combineBatch", "useMGroupReady", "rowStart", "rowEnd"):
        assert f"range.{field}" in production_combine
    zero_row_skip = production_combine.index("if (rowStart >= rowEnd)")
    group_wait = production_combine.index("WaitForGmm2MGroupReady(", zero_row_skip)
    assert zero_row_skip < group_wait
    assert "if (useMGroupReady && !rangeAlreadyReady)" in production_combine
    assert "} else if (!useMGroupReady)" in production_combine
    full_flag = production_combine.index("__gm__ int32_t *flagAddr =", group_wait)
    full_wait = production_combine.index("ReadGmByPassDCache(flagAddr)", full_flag)
    assert group_wait < full_flag < full_wait
    assert "useConsumerReadyGate" not in production_combine
    assert "WaitForGmm2ConsumerGate(" not in ready_wait_source
    assert "WaitForGmm2MGroupReadyThroughGate(" not in ready_wait_source
    assert "WriteGmByPassDCache(consumerGate" not in production_combine
    assert "dynamicClaim" not in production_combine
    assert "claimCursor" not in production_combine
    base = (root / "mc2/mega_moe/op_kernel/mega_moe_base.h").read_text(encoding="utf-8")
    assert "uint32_t GetRankStripedLogicalCore(" not in base
    assert "uint32_t GetCombineLogicalCore(" not in base
    assert "GetCoprimeCoreRotationStride" not in base
    assert "wideRankCoreCount" not in base
    dispatch_start = kernel.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchTokens("
    )
    dispatch_end = kernel.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::PrecomputeChainLen()",
        dispatch_start,
    )
    dispatch_source = kernel[dispatch_start:dispatch_end]
    assert "GetRankStripedLogicalCore(" not in dispatch_source
    assert "int32_t logicalWorker = static_cast<int32_t>(blockIdx_);" in dispatch_source
    pipeline_start = kernel.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCoreWavePipeline("
    )
    pipeline_end = kernel.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()",
        pipeline_start,
    )
    pipeline = kernel[pipeline_start:pipeline_end]
    combine_call = pipeline.index("bfloat16_t, true, false, false, true>")
    call_source = pipeline[combine_call : combine_call + 900]
    assert "GetMGroupsPerWave(), blockAivNum_, aivCoreIdx_, timer" in call_source


def test_legacy_token_owner_ready_sweep_model_and_source() -> None:
    ready_epoch = [[5, 5], [1], [2, 2], [3], []]
    chain_values = [[10, 11], [20], [30, 31], [40], []]
    processed = [False] * len(ready_epoch)
    processing_order: list[int] = []
    observed_chains: dict[int, list[int]] = {}
    for epoch in range(0, 6):
        for token, contribution_epochs in enumerate(ready_epoch):
            if processed[token]:
                continue
            if all(value <= epoch for value in contribution_epochs):
                observed_chains[token] = list(chain_values[token])
                processed[token] = True
                processing_order.append(token)
    assert processed == [True] * len(processed)
    assert processing_order == [4, 1, 2, 3, 0]
    assert observed_chains == {
        token: values for token, values in enumerate(chain_values)
    }
    # A sequential token-0-first core would expose none of tokens 1..3 until
    # epoch 5; the ready sweep completes each as soon as its own chain is ready.
    assert processing_order.index(1) < processing_order.index(0)

    combine = (ROOT / "mc2/mega_moe/op_kernel/mega_moe_combine_send.h").read_text(
        encoding="utf-8"
    )
    # This legacy helper is not instantiated by the production kernel. The
    # model and checks below cover only the retained helper's algorithm.
    kernel = (ROOT / "mc2/mega_moe/op_kernel/mega_moe.h").read_text(encoding="utf-8")
    assert "CombineTokensByOwner<" not in kernel
    owner_start = combine.index("void CombineTokensByOwner(")
    owner_end = combine.index("}  // namespace MegaMoeCombineImpl", owner_start)
    owner = combine[owner_start:owner_end]
    assert "while (!allTokensComplete)" in owner
    assert "IsTokenOwnerChainReady<EnableMGroupReady>" in owner
    assert "pendingCount == completedCount" in owner
    assert "if (completedCount > 0U)" in owner
    assert "idlePollMisses = 0U;" in owner
    assert "idlePollDelayCycles = 100;" in owner
    assert "} else if (!allTokensComplete)" in owner
    assert "GmSpinBackoffStep(idlePollMisses, idlePollDelayCycles);" in owner
    assert "tokenProcessed + tokenIdx" in owner
    assert "for (int32_t chainIdx = 0; chainIdx < chainLen; ++chainIdx)" in owner
    assert "LocalTensor<int32_t> expertPlan" in owner
    assert "DataCopyPad(expertPlan, expertPlanGm" in owner
    assert "SyncFunc<AscendC::HardEvent::MTE2_S>();" in owner
    assert "expertPlan.GetValue(expertIdx)" in owner
    # A remote output store owns its final accumulator row until MTE3 has
    # completed.  Publish tokenProcessed only after the ownership fence; the
    # former two-outstanding-event protocol was not device-correct on A5.
    output_copy = owner.index("DataCopyPad(gmRemoteD[outputOffset]")
    output_fence = owner.index(
        "SyncFunc<AscendC::HardEvent::MTE3_MTE2>();", output_copy
    )
    output_publish = owner.index("tokenProcessed + tokenIdx", output_fence)
    assert output_copy < output_fence < output_publish
    assert "outputPending0" not in owner
    assert "outputPending1" not in owner
    # Producer only emits M-group votes for a whole, unsliced expert.  Keep
    # the same wave-capacity guard in the ready precheck or Combine can wait
    # forever on a sliced expert that publishes only its full-expert flag.
    ready_start = combine.index("bool IsTokenOwnerChainReady(")
    ready_end = combine.index("template <typename ElementC", ready_start)
    ready = combine[ready_start:ready_end]
    assert "expertMGroups <= mGroupsPerWave" in ready


def test_expert_plan_header_is_built_in_allgather_and_reused() -> None:
    root = ROOT
    kernel = (root / "mc2/mega_moe/op_kernel/mega_moe.h").read_text(encoding="utf-8")
    workspace = (root / "mc2/mega_moe/op_kernel/mega_moe_workspace_info.h").read_text(
        encoding="utf-8"
    )

    order_begin = kernel.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::BuildExpertOrder()"
    )
    order_end = kernel.index(
        "void MegaMoe<TemplateMegaMoeTypeFunc>::LoadExpertScheduleMode()",
        order_begin,
    )
    order = kernel[order_begin:order_end]

    # The builder sources the raw per-rank window slices (already stable
    # after CrossRankSyncInWorldSize) through one batched UB copy per rank
    # instead of per-expert serial bypass reads of the reduced cumsum that
    # only the histogram barriers publish.  That independence is what allows
    # the build to run inside the AllGather phase.
    assert "expertTokenCumsum_" not in order
    assert "tokenPerExpertWin_[srcRank * expertNumAlign + sliceOffset]" in order
    assert order.count("DataCopyPad(planUb[stagingBase]") == 1
    assert "constexpr uint32_t int32PerDataBlock = 32U / sizeof(int32_t);" in order
    assert "stagingBaseUnaligned = 4U * expertPerRank_ + 1U" in order
    assert "int32PerDataBlock * int32PerDataBlock" in order
    for expert_per_rank in (3, 32, 64, 768):
        raw = 4 * expert_per_rank + 1
        staging = (raw + 7) // 8 * 8
        assert staging * 4 % 32 == 0
        assert staging >= raw
    assert "SyncFunc<AscendC::HardEvent::MTE3_S>();" in order
    assert "SyncFunc<AscendC::HardEvent::S_MTE2>();" in order
    assert "SyncFunc<AscendC::HardEvent::MTE2_S>();" in order
    assert "ReadGmByPassDCache" not in order

    # All four aggregates publish on both schedule paths: the header write
    # block sits before the early identity return.
    header_pub = order.index("uint32_t headerBase = 1U + 2U * expertPerRank_;")
    identity_return = order.index("WriteGmByPassDCache(selectedAddr, 0);")
    assert header_pub < identity_return
    for slot, value in (
        ("EXPERT_PLAN_HEADER_TOTAL_MGROUPS", "totalMGroups"),
        ("EXPERT_PLAN_HEADER_TOTAL_FULL_MGROUPS", "totalFullMGroups"),
        ("EXPERT_PLAN_HEADER_TOTAL_TAIL_ROWS", "totalTailRows"),
        ("EXPERT_PLAN_HEADER_MAX_ACTIVE_TILES", "maxActiveTiles"),
    ):
        assert order.count(slot) == 1
        write = re.search(
            rf"WriteGmByPassDCache\(selectedAddr\s*\+\s*headerBase\s*\+\s*{slot},"
            rf"\s*static_cast<int32_t>\({value}\)\);",
            order,
        )
        assert write is not None
        assert header_pub < write.start() < identity_return

    # Consumers: the AIC GMM1->GMM2 prescan reads two header slots and the
    # AIV fanout stride reads one; both keep the small-E scan fallback, so
    # E < EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS never dereferences a plan that
    # was never built.
    pipeline = kernel.index(
        "CoreWaveGmm2Plan MegaMoe<TemplateMegaMoeTypeFunc>::PrepareCoreWaveGmm2Plan("
    )
    prescan = kernel[
        pipeline : kernel.index(
            "void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2MnTiles(", pipeline
        )
    ]
    assert prescan.count("EXPERT_PLAN_HEADER_TOTAL_MGROUPS") == 1
    assert prescan.count("EXPERT_PLAN_HEADER_TOTAL_FULL_MGROUPS") == 1
    # Tail rows remain published, but the current prescan does not consume them.
    assert "EXPERT_PLAN_HEADER_TOTAL_TAIL_ROWS" not in prescan
    assert "expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS" in prescan
    assert "GetExpertRange(" in prescan
    fanout_begin = kernel.index("::DispatchFanoutWorkProfitable(")
    fanout_end = kernel.index("::ResetDispatchFanoutDescriptors()", fanout_begin)
    fanout = kernel[fanout_begin:fanout_end]
    assert fanout.count("EXPERT_PLAN_HEADER_MAX_ACTIVE_TILES") == 1
    assert "if (!priorityFromPlan)" in fanout
    assert "expertPerRank_ >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS" in fanout

    # Host and device share the single workspace reservation.
    assert "EXPERT_PLAN_HEADER_INTS = 4U" in workspace

    # Clamp parity: the builder aggregates must equal a GetExpertRange-style
    # rescan of the published endpoint table for any histogram and cap,
    # including caps that cut an expert mid-range or zero it out.
    rng = random.Random(950)
    for _ in range(300):
        expert_rows = [
            rng.choice((0, 0, rng.randint(1, 700))) for _ in range(rng.randint(3, 64))
        ]
        cap = rng.choice((1, 8, 256, 300, 349, 4096, 10**6))
        # Builder model: walks the accumulated cumsum exactly like the
        # kernel (localStart = current - rankBase - rows).
        b_groups = b_full = b_tail = 0
        b_stride = 1
        current = 0
        for rows in expert_rows:
            previous = current
            current += rows
            if rows <= 0:
                continue
            local_start = current - rows
            if local_start >= cap:
                continue
            clamped = min(rows, cap - local_start)
            if clamped <= 0:
                continue
            tiles = (clamped + 255) // 256
            b_groups += tiles
            b_full += clamped // 256
            b_tail += clamped % 256
            b_stride = max(b_stride, tiles)
        # GetExpertRange-style rescan of [start, end) endpoints.
        r_groups = r_full = r_tail = 0
        r_stride = 1
        endpoints = []
        acc = 0
        for rows in expert_rows:
            acc += rows
            endpoints.append(acc)
        start = 0
        for end in endpoints:
            rows = end - start
            local_start = start
            start = end
            if local_start < 0 or rows <= 0 or local_start >= cap:
                continue
            rows = min(rows, cap - local_start)
            if rows <= 0:
                continue
            tiles = (rows + 255) // 256
            r_groups += tiles
            r_full += rows // 256
            r_tail += rows % 256
            r_stride = max(r_stride, tiles)
        assert (b_groups, b_full, b_tail, b_stride) == (
            r_groups,
            r_full,
            r_tail,
            r_stride,
        )
