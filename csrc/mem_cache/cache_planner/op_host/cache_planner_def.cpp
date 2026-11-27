#include "../op_kernel/cache_planner_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"
#include <limits>

namespace optiling {
static ge::graphStatus TilingCachePlanner(gert::TilingContext *context)
{
    const auto *source = context->GetInputShape(0);
    const auto *topk = context->GetInputShape(3);
    const auto *attrs = context->GetAttrs();
    const auto *maxContext = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *capacity = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!source || !topk || !maxContext || !capacity || !context->GetPlatformInfo() || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(CachePlannerTilingData) || !workspace)
        return ge::GRAPH_FAILED;
    const auto &sourceShape = source->GetStorageShape();
    const auto &topkShape = topk->GetStorageShape();
    if (sourceShape.GetDimNum() != 2 || topkShape.GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto *slots = context->GetInputShape(1);
    const auto *requests = context->GetInputShape(2);
    if (!slots || !requests) return ge::GRAPH_FAILED;
    const auto &slotShape = slots->GetStorageShape();
    const auto &requestShape = requests->GetStorageShape();
    if (slotShape.GetDimNum() != 2 || requestShape.GetDimNum() != 1) return ge::GRAPH_FAILED;
    const int64_t rows = sourceShape.GetDim(0);
    const int64_t stride = sourceShape.GetDim(1);
    const int64_t batch = requestShape.GetDim(0);
    constexpr int64_t u32max = std::numeric_limits<uint32_t>::max();
    if (rows <= 1 || rows > u32max || stride <= 0 || stride > u32max || batch <= 0 ||
        batch > std::numeric_limits<int32_t>::max() || *maxContext <= 0 || *maxContext > (1LL << 24) ||
        stride < *maxContext || *capacity != 2048 || slotShape.GetDim(0) != rows || slotShape.GetDim(1) != *capacity ||
        topkShape.GetDim(0) != batch || topkShape.GetDim(1) != *capacity)
        return ge::GRAPH_FAILED;
    for (int i = 4; i < 10; ++i) {
        const auto *shape = context->GetInputShape(i);
        if (!shape) return ge::GRAPH_FAILED;
        const auto &dims = shape->GetStorageShape();
        int64_t elements = 1;
        for (size_t dim = 0; dim < dims.GetDimNum(); ++dim) {
            const int64_t size = dims.GetDim(dim);
            if (size <= 0 || elements > std::numeric_limits<int64_t>::max() / size) return ge::GRAPH_FAILED;
            elements *= size;
        }
        const int64_t expected = (i == 6 || i == 9) ? batch : batch * *capacity;
        if (elements != expected) return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<CachePlannerTilingData>();
    if (!data) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    auto cores = platform.GetCoreNumAiv();
    if (cores == 0) return ge::GRAPH_FAILED;
    data->realReqCount = static_cast<uint32_t>(sourceShape.GetDim(0) - 1);
    data->maxContextLen = static_cast<uint32_t>(*maxContext);
    data->sourceStride = static_cast<uint32_t>(sourceShape.GetDim(1));
    data->cacheCapacity = static_cast<uint32_t>(*capacity);
    data->batchSize = static_cast<uint32_t>(topkShape.GetDim(0));
    raw->SetDataSize(sizeof(*data));
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferShapeCachePlanner(gert::InferShapeContext *context)
{
    constexpr int indices[] = {0, 1, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 8; ++i) {
        const auto *input = context->GetInputShape(indices[i]);
        auto *output = context->GetOutputShape(i);
        if (!input || !output) return ge::GRAPH_FAILED;
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferTypeCachePlanner(gert::InferDataTypeContext *context)
{
    for (int i = 0; i < 8; ++i)
        context->SetOutputDataType(i, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
namespace ops {
class SglPlanSparseKvCache : public OpDef
{
public:
    explicit SglPlanSparseKvCache(const char *name) : OpDef(name)
    {
        for (const auto *input :
             {"source_to_slot", "slot_to_source", "req_ids", "topk_input", "topk_src_ids", "topk_dst_slots",
              "topk_miss_counts", "miss_src_ids", "miss_dst_slots", "miss_counts"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const auto *output : {"source_to_slot", "slot_to_source", "topk_src_ids", "topk_dst_slots",
                                   "topk_miss_counts", "miss_src_ids", "miss_dst_slots", "miss_counts"}) {
            this->Output(output)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Attr("max_context_len").AttrType(REQUIRED).Int();
        this->Attr("cache_capacity").AttrType(REQUIRED).Int();
        this->SetInferShape(optiling::InferShapeCachePlanner).SetInferDataType(optiling::InferTypeCachePlanner);
        this->AICore().SetTiling(optiling::TilingCachePlanner);
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglPlanSparseKvCache);
}  // namespace ops
