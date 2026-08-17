#include "mlp_lightning_indexer_tiling.h"

#include <string>
#include <unordered_map>

#include "../mlp_lightning_indexer_def.h"
#include "../../op_kernel/lightning_indexer_template_tiling_key.h"

namespace sglang::MlpLIHost {
namespace {

#define OPS_LOG_E(opName, logInfo) (std::string(opName) + ": " + logInfo)

}  // namespace

ge::graphStatus LIInfoParser::CheckRequiredParaExistence() const
{
    TORCH_CHECK(opParamInfo_.query.shape != nullptr, OPS_LOG_E(opName_, "query shape is nullptr"));
    TORCH_CHECK(opParamInfo_.query.desc != nullptr, OPS_LOG_E(opName_, "query desc is nullptr"));
    TORCH_CHECK(opParamInfo_.key.shape != nullptr, OPS_LOG_E(opName_, "key shape is nullptr"));
    TORCH_CHECK(opParamInfo_.key.desc != nullptr, OPS_LOG_E(opName_, "key desc is nullptr"));
    TORCH_CHECK(opParamInfo_.weights.shape != nullptr, OPS_LOG_E(opName_, "weights shape is nullptr"));
    TORCH_CHECK(opParamInfo_.weights.desc != nullptr, OPS_LOG_E(opName_, "weights desc is nullptr"));
    TORCH_CHECK(opParamInfo_.indicesOut.shape != nullptr, OPS_LOG_E(opName_, "indices output shape is nullptr"));
    TORCH_CHECK(opParamInfo_.indicesOut.desc != nullptr, OPS_LOG_E(opName_, "indices output desc is nullptr"));
    TORCH_CHECK(opParamInfo_.valuesOut.shape != nullptr, OPS_LOG_E(opName_, "values output shape is nullptr"));
    TORCH_CHECK(opParamInfo_.valuesOut.desc != nullptr, OPS_LOG_E(opName_, "values output desc is nullptr"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetOpName()
{
    TORCH_CHECK(context_ != nullptr, "TilingContext is null");
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetNpuInfo()
{
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    TORCH_CHECK(aivNum != 0 && aicNum != 0, OPS_LOG_E(opName_, "num of core obtained is 0"));
    socVersion_ = ascendcPlatform.GetSocVersion();
    TORCH_CHECK(socVersion_ == platform_ascendc::SocVersion::ASCEND910B ||
                    socVersion_ == platform_ascendc::SocVersion::ASCEND910_93,
                OPS_LOG_E(opName_, "soc version does not support"));
    TORCH_CHECK(context_->GetWorkspaceSizes(1) != nullptr, OPS_LOG_E(opName_, "workspaceSize is nullptr"));
    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    opParamInfo_.curSeqLengthsQ.tensor = context_->GetOptionalInputTensor(CUR_SEQ_Q_INDEX);
    opParamInfo_.curSeqLengthsQ.desc = context_->GetOptionalInputDesc(CUR_SEQ_Q_INDEX);
    opParamInfo_.curSeqLengths.tensor = context_->GetOptionalInputTensor(CUR_SEQ_K_INDEX);
    opParamInfo_.curSeqLengths.desc = context_->GetOptionalInputDesc(CUR_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.initTensor.tensor = context_->GetOptionalInputTensor(INIT_TENSOR_INDEX);
    opParamInfo_.initTensor.desc = context_->GetOptionalInputDesc(INIT_TENSOR_INDEX);
    opParamInfo_.localTensor.tensor = context_->GetOptionalInputTensor(LOCAL_TENSOR_INDEX);
    opParamInfo_.localTensor.desc = context_->GetOptionalInputDesc(LOCAL_TENSOR_INDEX);
}

void LIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.indicesOut.desc = context_->GetOutputDesc(SPARSE_INDICES_INDEX);
    opParamInfo_.indicesOut.shape = context_->GetOutputShape(SPARSE_INDICES_INDEX);
    opParamInfo_.valuesOut.desc = context_->GetOutputDesc(SPARSE_VALUES_INDEX);
    opParamInfo_.valuesOut.shape = context_->GetOutputShape(SPARSE_VALUES_INDEX);
}

ge::graphStatus LIInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    TORCH_CHECK(attrs != nullptr, OPS_LOG_E(opName_, "attrs is nullptr"));
    const char *layoutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    const char *layoutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    auto sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    auto blockLen = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_LEN_INDEX);
    auto qBlockLen = attrs->GetAttrPointer<int32_t>(ATTR_Q_BLOCK_LEN_INDEX);
    auto initNum = attrs->GetAttrPointer<int32_t>(ATTR_INIT_NUM_INDEX);
    auto localNum = attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_NUM_INDEX);
    auto sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    auto preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    auto nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    auto returnValue = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
    TORCH_CHECK(layoutQuery != nullptr && layoutKey != nullptr, OPS_LOG_E(opName_, "layout attrs are nullptr"));
    TORCH_CHECK(sparseCount != nullptr && blockLen != nullptr && qBlockLen != nullptr && initNum != nullptr &&
                    localNum != nullptr && sparseMode != nullptr && preTokens != nullptr &&
                    nextTokens != nullptr && returnValue != nullptr,
                OPS_LOG_E(opName_, "numeric attrs are nullptr"));

    opParamInfo_.indicesOut.desc = opParamInfo_.indicesOut.desc;

    TORCH_CHECK((std::string(layoutKey) == "PA_BSND") || (std::string(layoutQuery) == std::string(layoutKey)),
                OPS_LOG_E(opName_, "under non-PA conditions, layout_query and layout_key should be equal"));
    TORCH_CHECK(std::string(layoutQuery) == "BSND" || std::string(layoutQuery) == "TND",
                OPS_LOG_E(opName_, "layout_query only supports BSND or TND"));
    TORCH_CHECK(std::string(layoutKey) == "PA_BSND" || std::string(layoutKey) == "BSND" ||
                    std::string(layoutKey) == "TND",
                OPS_LOG_E(opName_, "layout_key only supports PA_BSND, BSND or TND"));
    TORCH_CHECK(*blockLen == 1 || *blockLen == 2 || *blockLen == 4 || *blockLen == 8 || *blockLen == 16,
                OPS_LOG_E(opName_, "kv_block_len must be in [1, 2, 4, 8, 16]"));
    TORCH_CHECK(*qBlockLen == 1, OPS_LOG_E(opName_, "q_block_len must be 1"));
    TORCH_CHECK(((*sparseCount > 0) && (*sparseCount <= static_cast<int32_t>(TOPK_2K))) ||
                    ((*sparseCount <= static_cast<int32_t>(TOPK_8K)) && (*sparseCount % 128 == 0)),
                OPS_LOG_E(opName_, "sparse_count must be in [1, 2048] or divisible by 128 and <= 8192"));
    TORCH_CHECK(*initNum >= 0 && *initNum <= 16 && (*initNum % *blockLen == 0),
                OPS_LOG_E(opName_, "init_num is invalid"));
    TORCH_CHECK(*localNum >= 0 && *localNum <= 2048 && (*localNum % *blockLen == 0),
                OPS_LOG_E(opName_, "local_num is invalid"));
    TORCH_CHECK(*sparseMode == 0 || *sparseMode == 3, OPS_LOG_E(opName_, "sparse_mode only supports 0 or 3"));
    TORCH_CHECK(!(*qBlockLen > 1 && (*initNum != 0 || *localNum != 0 || *blockLen > 1)),
                OPS_LOG_E(opName_, "q_block_len and kv_block_len do not support setting meanwhile"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.indicesOut.desc->GetDataType();
    valuesOutType_ = opParamInfo_.valuesOut.desc->GetDataType();
    TORCH_CHECK(inputQType_ == inputKType_, OPS_LOG_E(opName_, "query and key dtypes must match"));
    TORCH_CHECK(inputQType_ == ge::DT_FLOAT16 || inputQType_ == ge::DT_BF16,
                OPS_LOG_E(opName_, "query and key must be fp16 or bf16"));
    TORCH_CHECK(weightsType_ == ge::DT_FLOAT, OPS_LOG_E(opName_, "weights must be float32"));
    TORCH_CHECK(outputType_ == ge::DT_INT32, OPS_LOG_E(opName_, "sparse_indices must be int32"));
    TORCH_CHECK(valuesOutType_ == inputQType_, OPS_LOG_E(opName_, "sparse_values dtype must match query"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetQueryKeyAndOutLayout()
{
    auto attrs = context_->GetAttrs();
    std::string layout(attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX));
    std::string keyLayout(attrs->GetStr(ATTR_KEY_LAYOUT_INDEX));
    static const std::unordered_map<std::string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND},
        {"TND", DataLayout::TND},
        {"PA_BSND", DataLayout::PA_BSND},
    };
    qLayout_ = layoutMap.at(layout);
    kLayout_ = layoutMap.at(keyLayout);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::PA_BSND) {
        TORCH_CHECK(opParamInfo_.blockTable.tensor != nullptr, OPS_LOG_E(opName_, "block_table must not be null"));
        TORCH_CHECK(opParamInfo_.curSeqLengths.tensor != nullptr,
                    OPS_LOG_E(opName_, "cur_seq_lengths_key must not be null"));
        TORCH_CHECK(opParamInfo_.blockTable.desc->GetDataType() == ge::DT_INT32,
                    OPS_LOG_E(opName_, "block_table only supports int32"));
    } else if (kLayout_ == DataLayout::TND) {
        TORCH_CHECK(opParamInfo_.curSeqLengths.tensor != nullptr,
                    OPS_LOG_E(opName_, "cur_seq_lengths_key must not be null"));
    }
    TORCH_CHECK(opParamInfo_.curSeqLengths.tensor == nullptr ||
                    opParamInfo_.curSeqLengths.desc->GetDataType() == ge::DT_INT64,
                OPS_LOG_E(opName_, "cur_seq_lengths_key only supports int64"));
    if (qLayout_ == DataLayout::TND) {
        TORCH_CHECK(opParamInfo_.curSeqLengthsQ.tensor != nullptr,
                    OPS_LOG_E(opName_, "cur_seq_lengths_query must not be null"));
    }
    TORCH_CHECK(opParamInfo_.curSeqLengthsQ.tensor == nullptr ||
                    opParamInfo_.curSeqLengthsQ.desc->GetDataType() == ge::DT_INT64,
                OPS_LOG_E(opName_, "cur_seq_lengths_query only supports int64"));
    TORCH_CHECK(kLayout_ == DataLayout::PA_BSND || opParamInfo_.blockTable.tensor == nullptr,
                OPS_LOG_E(opName_, "block_table must be null when layout_key is not PA_BSND"));
    TORCH_CHECK(opParamInfo_.initTensor.tensor == nullptr || opParamInfo_.initTensor.desc->GetDataType() == ge::DT_INT32,
                OPS_LOG_E(opName_, "init_tensor only supports int32"));
    TORCH_CHECK(opParamInfo_.localTensor.tensor == nullptr || opParamInfo_.localTensor.desc->GetDataType() == ge::DT_INT32,
                OPS_LOG_E(opName_, "local_tensor only supports int32"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckShapeDim()
{
    if (opParamInfo_.blockTable.tensor != nullptr) {
        TORCH_CHECK(opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() == DIM_NUM_TWO,
                    OPS_LOG_E(opName_, "block_table must be 2D"));
    }
    uint32_t qExpectShapeDim = qLayout_ == DataLayout::TND ? DIM_NUM_THREE : DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = kLayout_ == DataLayout::TND ? DIM_NUM_THREE : DIM_NUM_FOUR;
    TORCH_CHECK(opParamInfo_.query.shape->GetStorageShape().GetDimNum() == qExpectShapeDim,
                OPS_LOG_E(opName_, "query dim num is invalid"));
    TORCH_CHECK(opParamInfo_.key.shape->GetStorageShape().GetDimNum() == kExpectShapeDim,
                OPS_LOG_E(opName_, "key dim num is invalid"));
    TORCH_CHECK(opParamInfo_.weights.shape->GetStorageShape().GetDimNum() == qExpectShapeDim - 1,
                OPS_LOG_E(opName_, "weights dim num is invalid"));
    TORCH_CHECK(opParamInfo_.indicesOut.shape->GetStorageShape().GetDimNum() == qExpectShapeDim,
                OPS_LOG_E(opName_, "indices output dim num is invalid"));
    TORCH_CHECK(opParamInfo_.valuesOut.shape->GetStorageShape().GetDimNum() == qExpectShapeDim,
                OPS_LOG_E(opName_, "values output dim num is invalid"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetN1Size()
{
    n1Size_ = qLayout_ == DataLayout::BSND
                  ? static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO))
                  : static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetCurSeqLenSize(uint32_t &size, const gert::Tensor *tensor, const std::string &name)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize()) - 1;
    TORCH_CHECK(size > 0, name, "'s shape size should be greater than 1");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    TORCH_CHECK(n2Size_ == 1, OPS_LOG_E(opName_, "key head num only supports 1"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetGSize()
{
    TORCH_CHECK(n1Size_ % n2Size_ == 0, OPS_LOG_E(opName_, "query head_num can not be a multiple of key head_num"));
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetBatchSize()
{
    if (qLayout_ == DataLayout::TND) {
        return GetCurSeqLenSize(bSize_, opParamInfo_.curSeqLengthsQ.tensor, "cur_seq_lengths_query");
    }
    bSize_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetHeadDim()
{
    uint32_t dIndex = qLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    headDim_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex));
    TORCH_CHECK(headDim_ == HEAD_DIM_LIMIT, OPS_LOG_E(opName_, "head_dim only supports 128"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    TORCH_CHECK(blockSize_ % 16 == 0 && blockSize_ > 0 && blockSize_ <= 1024,
                OPS_LOG_E(opName_, "key block size must be a multiple of 16 and in (0, 1024]"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckBlockCount()
{
    auto blockCount = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    TORCH_CHECK(blockCount != 0, OPS_LOG_E(opName_, "key block_count cannot be 0"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS2SizeForPageAttention()
{
    GetAndCheckBlockSize();
    CheckBlockCount();
    maxBlockNumPerBatch_ = static_cast<uint32_t>(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS2Size()
{
    if (kLayout_ == DataLayout::PA_BSND) {
        return GetS2SizeForPageAttention();
    }
    if (kLayout_ == DataLayout::TND) {
        s2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    } else {
        s2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::ValidateInputShapesMatch()
{
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        TORCH_CHECK(opParamInfo_.curSeqLengths.tensor->GetShapeSize() - 1 == bSize_,
                    OPS_LOG_E(opName_, "cur_seq_lengths_query and cur_seq_lengths_key batch size mismatch"));
        TORCH_CHECK(opParamInfo_.blockTable.tensor == nullptr ||
                        opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) == bSize_,
                    OPS_LOG_E(opName_, "block_table batch size mismatch"));
        uint32_t qTSize = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(0));
        TORCH_CHECK(opParamInfo_.weights.shape->GetStorageShape().GetDim(0) == qTSize &&
                        opParamInfo_.indicesOut.shape->GetStorageShape().GetDim(0) == qTSize &&
                        opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) == qTSize,
                    OPS_LOG_E(opName_, "query/weights/outputs T dim mismatch"));
    } else {
        TORCH_CHECK(opParamInfo_.weights.shape->GetStorageShape().GetDim(0) == bSize_,
                    OPS_LOG_E(opName_, "weights batch size mismatch"));
        TORCH_CHECK(opParamInfo_.blockTable.tensor == nullptr ||
                        opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) == bSize_,
                    OPS_LOG_E(opName_, "block_table batch size mismatch"));
        TORCH_CHECK(opParamInfo_.curSeqLengths.tensor == nullptr ||
                        opParamInfo_.curSeqLengths.tensor->GetShapeSize() == bSize_,
                    OPS_LOG_E(opName_, "cur_seq_lengths_key batch size mismatch"));
        TORCH_CHECK(opParamInfo_.curSeqLengthsQ.tensor == nullptr ||
                        opParamInfo_.curSeqLengthsQ.tensor->GetShapeSize() == bSize_,
                    OPS_LOG_E(opName_, "cur_seq_lengths_query batch size mismatch"));
        TORCH_CHECK(opParamInfo_.indicesOut.shape->GetStorageShape().GetDim(0) == bSize_ &&
                        opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) == bSize_,
                    OPS_LOG_E(opName_, "output batch size mismatch"));
        TORCH_CHECK(opParamInfo_.weights.shape->GetStorageShape().GetDim(1) == s1Size_ &&
                        opParamInfo_.indicesOut.shape->GetStorageShape().GetDim(1) == s1Size_ &&
                        opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(1) == s1Size_,
                    OPS_LOG_E(opName_, "S1 dim mismatch"));
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }

    TORCH_CHECK(opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) == n1Size_,
                OPS_LOG_E(opName_, "N1 dim mismatch"));
    uint32_t keyDDim = kLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    TORCH_CHECK(opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) == headDim_,
                OPS_LOG_E(opName_, "D dim mismatch"));
    TORCH_CHECK(opParamInfo_.indicesOut.shape->GetStorageShape().GetDim(outN2Dim) == n2Size_ &&
                    opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim) == n2Size_,
                OPS_LOG_E(opName_, "N2 dim mismatch"));
    auto attrs = context_->GetAttrs();
    auto sparseCount = *attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    TORCH_CHECK(opParamInfo_.indicesOut.shape->GetStorageShape().GetDim(outN2Dim + 1) == sparseCount &&
                    opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim + 1) == sparseCount,
                OPS_LOG_E(opName_, "sparse_count dim mismatch"));
    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GenerateInfo(LITilingInfo &liInfo)
{
    auto attrs = context_->GetAttrs();
    liInfo.opName = opName_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;
    liInfo.bSize = bSize_;
    liInfo.n1Size = n1Size_;
    liInfo.n2Size = n2Size_;
    liInfo.s1Size = s1Size_;
    liInfo.s2Size = s2Size_;
    liInfo.gSize = gSize_;
    liInfo.headDim = headDim_;
    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.weightsType = weightsType_;
    liInfo.outputType = outputType_;
    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;
    liInfo.pageAttentionFlag = std::string(attrs->GetStr(ATTR_KEY_LAYOUT_INDEX)) == "PA_BSND";
    liInfo.sparseMode = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX));
    liInfo.sparseCount = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX));
    liInfo.blockLen = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_LEN_INDEX));
    liInfo.qBlockLen = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(ATTR_Q_BLOCK_LEN_INDEX));
    liInfo.initNum = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(ATTR_INIT_NUM_INDEX));
    liInfo.localNum = static_cast<uint32_t>(*attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_NUM_INDEX));
    liInfo.preTokens = *attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    liInfo.nextTokens = *attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    liInfo.returnValue = static_cast<int8_t>(*attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX));
    liInfo.inputQLayout = qLayout_;
    liInfo.inputKLayout = kLayout_;
}

ge::graphStatus LIInfoParser::ParseAndCheck(LITilingInfo &liInfo)
{
    GetOpName();
    GetNpuInfo();
    GetInputParaInfo();
    GetOutputParaInfo();
    CheckRequiredParaExistence();
    GetAndCheckAttrParaInfo();
    GetAndCheckInOutDataType();
    GetQueryKeyAndOutLayout();
    GetAndCheckOptionalInput();
    CheckShapeDim();
    GetN1Size();
    GetAndCheckN2Size();
    GetGSize();
    GetBatchSize();
    GetS1Size();
    GetHeadDim();
    GetS2Size();
    ValidateInputShapesMatch();
    GenerateInfo(liInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerTiling::DoTiling(LITilingInfo *tilingInfo)
{
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t M_BASE_SIZE = 512;
    constexpr uint32_t S2_BASE_SIZE = 512;
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8;
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;
    constexpr uint32_t S1_BASE_SIZE = 8;
    constexpr uint32_t TOPK_MAX_SIZE = 2048;
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    context_->SetWorkspaceSizes(workspaceSize - ascendcPlatform.GetLibApiWorkSpaceSize());

    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);

    tilingData_.bSize = tilingInfo->bSize;
    tilingData_.n2Size = tilingInfo->n2Size;
    tilingData_.gSize = tilingInfo->gSize;
    tilingData_.s1Size = tilingInfo->s1Size;
    tilingData_.s2Size = tilingInfo->s2Size;
    tilingData_.sparseCount = tilingInfo->sparseCount;
    tilingData_.blockLen = tilingInfo->blockLen;
    tilingData_.qBlockLen = tilingInfo->qBlockLen;
    tilingData_.initNum = tilingInfo->initNum;
    tilingData_.localNum = tilingInfo->localNum;
    tilingData_.usedCoreNum = blockDim;
    tilingData_.blockSize = tilingInfo->blockSize;
    tilingData_.maxBlockNumPerBatch = tilingInfo->maxBlockNumPerBatch;
    tilingData_.sparseMode = tilingInfo->sparseMode;
    tilingData_.preTokens = tilingInfo->preTokens;
    tilingData_.nextTokens = tilingInfo->nextTokens;
    tilingData_.returnValue = tilingInfo->returnValue;
    tilingData_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

}  // namespace sglang::MlpLIHost
