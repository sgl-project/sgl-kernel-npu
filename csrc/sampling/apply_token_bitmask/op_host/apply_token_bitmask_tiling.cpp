#include "apply_token_bitmask_plan.h"
#include "apply_token_bitmask_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingApplyTokenBitMask(gert::TilingContext* context)
{
    const auto* logits = context->GetInputShape(0);
    const auto* bitmask = context->GetInputShape(1);
    const auto* logitsDesc = context->GetInputDesc(0);
    const auto* bitmaskDesc = context->GetInputDesc(1);
    if (!logits || !bitmask || !logitsDesc || !bitmaskDesc || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto& logitsShape = logits->GetStorageShape();
    const auto& bitmaskShape = bitmask->GetStorageShape();
    // Validate the existing Kernel's working-tensor contract, not the public Torch input.
    if (logitsShape.GetDimNum() != 2 || bitmaskShape.GetDimNum() != 2 ||
        logitsShape.GetDim(0) < 0 || logitsShape.GetDim(0) != bitmaskShape.GetDim(0) ||
        logitsShape.GetDim(1) <= 0 || logitsShape.GetDim(1) % 256 != 0 ||
        bitmaskShape.GetDim(1) < logitsShape.GetDim(1) / 32 || bitmaskDesc->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    int64_t dtypeSize = 0;
    switch (logitsDesc->GetDataType()) {
        case ge::DT_FLOAT: dtypeSize = 4; break;
        case ge::DT_FLOAT16:
        case ge::DT_BF16: dtypeSize = 2; break;
        default: return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const auto plan = sgl::MakeBitmaskPlan(logitsShape.GetDim(0), logitsShape.GetDim(1),
                                         bitmaskShape.GetDim(1), dtypeSize, platform.GetCoreNumAiv(), ubSize);
    SglApplyTokenBitMaskTilingData tiling;
    tiling.set_numRows(plan.numRows);
    tiling.set_vocabSize(plan.vocabSize);
    tiling.set_logitsStride(plan.logitsStride);
    tiling.set_bitmaskStride(plan.bitmaskStride);
    tiling.set_baseRows(plan.baseRows);
    tiling.set_extraCores(plan.extraCores);
    tiling.set_tileLength(plan.tileLength);
    tiling.set_blockDim(plan.blockDim);
    tiling.set_dtypeSize(plan.dtypeSize);

    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < tiling.GetDataSize() || !workspace) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(plan.blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SglApplyTokenBitMask).Tiling(TilingApplyTokenBitMask);

}  // namespace optiling
