#include "sgl_operator_config.h"
#include "../op_kernel/cache_loc_tiling.h"
#include "../../../utils/common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// Compute scheduling and buffer sizes in CANN-managed tiling storage.
inline ge::graphStatus BuildTiling(gert::TilingContext *context, bool is_update)
{
    const auto *req = context->GetInputShape(0);
    const auto *pool = context->GetInputShape(1);
    const auto *locations = context->GetInputShape(4);
    const auto *reqDesc = context->GetInputDesc(0);
    if (!req || !pool || !locations || !reqDesc || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &reqShape = req->GetOriginShape();
    const auto &poolShape = pool->GetOriginShape();
    const auto &locShape = locations->GetOriginShape();
    if (reqShape.GetDimNum() < 1 || poolShape.GetDimNum() < 2 || reqShape.GetDim(0) < 0 || poolShape.GetDim(0) < 0 ||
        poolShape.GetDim(1) < 0 || locShape.GetShapeSize() < 0) {
        return ge::GRAPH_FAILED;
    }
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglCacheLocTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tillingData = context->GetTilingData<SglCacheLocTilingData>();
    if (!tillingData) return ge::GRAPH_FAILED;

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t blockDim = is_update ? 1 : platform.GetCoreNumAiv();
    if (blockDim == 0) return ge::GRAPH_FAILED;
    const uint64_t batchSize = reqShape.GetDim(0);
    tillingData->vcoreNum = blockDim;
    tillingData->poolSize = poolShape.GetDim(0);
    tillingData->batchSize = batchSize;
    tillingData->rowNumNoTail = batchSize / tillingData->vcoreNum;
    tillingData->tailNum = batchSize % tillingData->vcoreNum;
    tillingData->rowSize = poolShape.GetDim(1);
    if (reqDesc->GetDataType() == ge::DT_INT32) {
        tillingData->key = 1;
        tillingData->reqInxBufferCount = host_utils::alinInt32Count(batchSize);
        tillingData->reqInxBufferSize = tillingData->reqInxBufferCount * sizeof(int32_t);
    } else if (reqDesc->GetDataType() == ge::DT_INT64) {
        tillingData->key = 2;
        tillingData->reqInxBufferCount = host_utils::alinInt64Count(batchSize);
        tillingData->reqInxBufferSize = tillingData->reqInxBufferCount * sizeof(int64_t);
    } else {
        return ge::GRAPH_FAILED;
    }
    tillingData->tokenCountAlignInt32 = host_utils::alinInt32Count(MAX_STEP);
    tillingData->tokenColAlignInt32 = tillingData->tokenCountAlignInt32 * sizeof(int32_t);
    tillingData->offsetCountAlignInt64 = host_utils::alinInt64Count(batchSize);
    tillingData->offsetColAlignInt64 = tillingData->offsetCountAlignInt64 * sizeof(int64_t);
    // Use the logical view length, never batch * MAX_STEP or the backing allocation.
    tillingData->cacheLocSize = locShape.GetShapeSize();
    tillingData->cacheLocCountAlignInt32 = host_utils::alinInt32Count(tillingData->cacheLocSize);
    tillingData->cacheLocAlignInt32 = tillingData->cacheLocCountAlignInt32 * sizeof(int32_t);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t ubBufferSizeToUse = tillingData->tokenColAlignInt32 + 3 * tillingData->offsetColAlignInt64 +
                                 3 * batchSize * sizeof(int32_t) + tillingData->cacheLocAlignInt32;
    if (ubBufferSizeToUse > ubSize) return ge::GRAPH_FAILED;
    raw->SetDataSize(sizeof(SglCacheLocTilingData));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

template <int Input>
inline ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(Input);
    auto *output = context->GetOutputShape(0);
    if (!input || !output) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

template <int Input>
inline ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(Input));
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace optiling {
ge::graphStatus TilingAssign(gert::TilingContext *context)
{
    return optiling::BuildTiling(context, false);
}
}  // namespace optiling

namespace ops {
class SglCacheLocAssign : public OpDef
{
public:
    explicit SglCacheLocAssign(const char *name) : OpDef(name)
    {
        this->Input("req_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("token_pool")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("start_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("out_cache_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // A same-name output generates a Ref argument instead of a new allocation.
        this->Output("token_pool")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(optiling::InferShape<1>).SetInferDataType(optiling::InferDataType<1>);
        this->AICore().SetTiling(optiling::TilingAssign);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglCacheLocAssign);
}  // namespace ops


namespace optiling {
ge::graphStatus TilingUpdate(gert::TilingContext *context)
{
    return optiling::BuildTiling(context, true);
}
}  // namespace optiling

namespace ops {
class SglCacheLocUpdate : public OpDef
{
public:
    explicit SglCacheLocUpdate(const char *name) : OpDef(name)
    {
        this->Input("req_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("token_pool")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("start_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("out_cache_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // A same-name output generates a Ref argument instead of a new allocation.
        this->Output("out_cache_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(optiling::InferShape<4>).SetInferDataType(optiling::InferDataType<4>);
        this->AICore().SetTiling(optiling::TilingUpdate);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglCacheLocUpdate);
}  // namespace ops
