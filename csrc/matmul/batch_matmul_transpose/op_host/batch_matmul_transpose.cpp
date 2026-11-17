#include <cstring>
#include <initializer_list>
#include <limits>
#include "tiling/tiling_data.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingBatchMatmulTranspose(gert::TilingContext *context)
{
    if (!context) return ge::GRAPH_FAILED;
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    const auto *aDesc = context->GetInputDesc(0);
    const auto *bDesc = context->GetInputDesc(1);
    if (!a || !b || !aDesc || !bDesc) return ge::GRAPH_FAILED;
    const auto *attrs = context->GetAttrs();
    const auto *format = attrs ? attrs->GetAttrPointer<char>(0) : nullptr;
    const auto *quant = attrs ? attrs->GetAttrPointer<char>(1) : nullptr;
    const bool nz = format && std::strcmp(format, "NZ") == 0;
    if (format && !nz && std::strcmp(format, "ND") != 0) return ge::GRAPH_FAILED;
    uint32_t quantMode = 0;
    if (quant && std::strcmp(quant, "per_channel_symm") != 0) {
        if (std::strcmp(quant, "per_channel_asymm") == 0)
            quantMode = 1;
        else if (std::strcmp(quant, "per_token_symm") == 0)
            quantMode = 2;
        else
            return ge::GRAPH_FAILED;
    }
    const auto &as = a->GetStorageShape();
    const auto &bs = b->GetStorageShape();
    if (as.GetDimNum() != 3 || bs.GetDimNum() != (nz ? 4 : 3)) return ge::GRAPH_FAILED;
    constexpr uint64_t max = std::numeric_limits<uint32_t>::max();
    for (const auto *shape : {&as, &bs}) {
        uint64_t elements = 1;
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            const auto dim = shape->GetDim(i);
            if (dim <= 0 || static_cast<uint64_t>(dim) > max / elements) return ge::GRAPH_FAILED;
            elements *= dim;
        }
    }
    const int64_t m = as.GetDim(0), batch = as.GetDim(1), k = as.GetDim(2);
    const int64_t n = nz ? bs.GetDim(1) * bs.GetDim(3) : bs.GetDim(2);
    if (m > 1024 || batch != bs.GetDim(0) || k != bs.GetDim(nz ? 2 : 1) ||
        (nz && (bs.GetDim(3) != 16 || k % 16 != 0)) || static_cast<uint64_t>(m) * batch > max / n)
        return ge::GRAPH_FAILED;
    const auto dtype = aDesc->GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) || bDesc->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    optiling::HardwareInfo hw;
    optiling::MatMulInfo info;
    info.batchSize = static_cast<uint32_t>(batch);
    info.m = static_cast<uint32_t>(m);
    info.k = static_cast<uint32_t>(k);
    info.n = static_cast<uint32_t>(n);
    info.dtypeA = info.dtypeB = info.dtypeC =
        dtype == ge::DT_BF16 ? optiling::TensorDType::TENSOR_DTYPE_BF16 : optiling::TensorDType::TENSOR_DTYPE_FLOAT16;
    info.formatB = nz ? optiling::TensorFormat::TENSOR_FORMAT_NZ : optiling::TensorFormat::TENSOR_FORMAT_ND;
    info.mmType = optiling::MatMul::MatMulType::MATMUL_EIN_SUM;
    info.inDtype = info.outDtype = 2.0f;
    info.quantMode = static_cast<optiling::MatMul::QuantMode>(quantMode);
    optiling::BatchMatmulTransposeTilingData data;
    uint32_t blockDim = 0;
    optiling::TilingBatchMatmulTranspose(info, hw, blockDim, data);
    if (!data.opShape.m0 || !data.opShape.k0 || !data.opShape.n0 || !data.mLoop || !data.kLoop || !data.nLoop ||
        !blockDim)
        return ge::GRAPH_FAILED;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(data) || !workspace) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);  // The unchanged kernel dispatches using data.tilingKey.
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {

class SglBatchMatmulTranspose : public OpDef
{
public:
    explicit SglBatchMatmulTranspose(const char *name) : OpDef(name)
    {
        for (const auto *input : {"tensor_a", "tensor_b"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        // NZ is explicitly packed [B, N/16, K, 16] storage in an ND descriptor.
        this->Output("tensor_c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("format_mode").AttrType(OPTIONAL).String("ND");
        this->Attr("quant_mode").AttrType(OPTIONAL).String("per_channel_symm");
        this->AICore().SetTiling(
            static_cast<ge::graphStatus (*)(gert::TilingContext *)>(optiling::TilingBatchMatmulTranspose));
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SglBatchMatmulTranspose);
}  // namespace ops
