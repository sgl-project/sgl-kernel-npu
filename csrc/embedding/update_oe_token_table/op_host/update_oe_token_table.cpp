#include "sgl_operator_config.h"
#include <type_traits>

#include "update_oe_token_table.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFEER = 2;
constexpr int64_t TILING_KEY = 10000;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024;

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }
    return x;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type CeilAlign(T x, T align)
{
    return CeilDiv(x, align) * align;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type FloorAlign(T x, T align)
{
    return align == 0 ? 0 : x / align * align;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    UpdateOeTokenTableTilingData tiling;
    auto batch_size = *(context->GetAttrs()->GetAttrPointer<int64_t>(0));
    auto max_context_len = *(context->GetAttrs()->GetAttrPointer<int64_t>(1));
    auto ignoreTokenShape = context->GetInputShape(4)->GetStorageShape();  // ignore_tokens is input 4
    auto ignore_token_num = ignoreTokenShape.GetDim(0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto blockFactor = CeilDiv(static_cast<uint32_t>(batch_size), aivNum);
    auto usedCoreNum = CeilDiv(static_cast<uint32_t>(batch_size), blockFactor);
    blockFactor = CeilDiv(static_cast<uint32_t>(batch_size), usedCoreNum);
    auto tailBlockFactor = batch_size % usedCoreNum == 0 ? blockFactor : batch_size % blockFactor;

    int64_t ignoreUbSize = CeilAlign(static_cast<int64_t>(ignore_token_num * sizeof(int32_t)), BLOCK_SIZE);
    int64_t reqLenUbSize = CeilAlign(static_cast<int64_t>(batch_size * sizeof(int32_t)), BLOCK_SIZE);
    int64_t reserveUbSize = ubSize - ignoreUbSize - reqLenUbSize;
    int32_t ubFactor =
        FloorAlign(reserveUbSize / (DOUBLE_BUFEER * 4), BLOCK_SIZE) / sizeof(int32_t);  // 4: 计算需要4个tensor

    context->SetBlockDim(usedCoreNum);
    tiling.set_batchSize(batch_size);
    tiling.set_blockFactor(blockFactor);
    tiling.set_tailBlockFactor(tailBlockFactor);
    tiling.set_ubFactor(ubFactor);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_maxContextLen(max_context_len);
    tiling.set_ignoreTokenNum(ignore_token_num);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(TILING_KEY);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SglUpdateOeTokenTable : public OpDef
{
public:
    explicit SglUpdateOeTokenTable(const char *name) : OpDef(name)
    {
        this->Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("req_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("row_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("column_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ignore_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("oe_token_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batch_size").Int();
        this->Attr("max_context_len").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglUpdateOeTokenTable);
}  // namespace ops
