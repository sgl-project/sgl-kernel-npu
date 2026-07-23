#ifndef MLP_LIGHTNING_INDEXER_TILING_H
#define MLP_LIGHTNING_INDEXER_TILING_H

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ge_helper.h"
#include "mlp_lightning_tiling_data.h"

namespace sglang::MlpLIHost {

struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t { BSND = 0, TND = 1, PA_BSND = 2 };

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHTS_INDEX = 2;
constexpr uint32_t CUR_SEQ_Q_INDEX = 3;
constexpr uint32_t CUR_SEQ_K_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t INIT_TENSOR_INDEX = 6;
constexpr uint32_t LOCAL_TENSOR_INDEX = 7;
constexpr uint32_t SPARSE_INDICES_INDEX = 0;
constexpr uint32_t SPARSE_VALUES_INDEX = 1;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t TOPK_2K = 2048;
constexpr uint32_t TOPK_8K = 8192;

struct LiParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingOptionalParaInfo curSeqLengthsQ = {nullptr, nullptr};
    TilingOptionalParaInfo curSeqLengths = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingOptionalParaInfo initTensor = {nullptr, nullptr};
    TilingOptionalParaInfo localTensor = {nullptr, nullptr};
    TilingRequiredParaInfo indicesOut = {nullptr, nullptr};
    TilingRequiredParaInfo valuesOut = {nullptr, nullptr};
};

class LITilingInfo {
public:
    const char *opName = nullptr;
    LiParaInfo opParamInfo;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t headDim = 0;
    uint32_t gSize = 0;
    bool pageAttentionFlag = false;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t sparseMode = 0;
    uint32_t sparseCount = 0;
    uint32_t blockLen = 1;
    uint32_t qBlockLen = 1;
    uint32_t initNum = 0;
    uint32_t localNum = 0;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int8_t returnValue = 0;
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType weightsType = ge::DT_FLOAT;
    ge::DataType outputType = ge::DT_INT32;
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BSND;
};

class LIInfoParser {
public:
    explicit LIInfoParser(ge_helper::TilingContext *context) : context_(context) {}

    ge::graphStatus ParseAndCheck(LITilingInfo &liInfo);

private:
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetCurSeqLenSize(uint32_t &size, const gert::Tensor *tensor, const std::string &name);
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus ValidateInputShapesMatch();
    void GenerateInfo(LITilingInfo &liInfo);

    ge_helper::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    LiParaInfo opParamInfo_;
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    uint32_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    DataLayout qLayout_ = DataLayout::BSND;
    DataLayout kLayout_ = DataLayout::PA_BSND;
    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT;
    ge::DataType outputType_ = ge::DT_INT32;
    ge::DataType valuesOutType_ = ge::DT_FLOAT16;
};

class LightningIndexerTiling {
public:
    explicit LightningIndexerTiling(ge_helper::TilingContext *context) : context_(context) {}
    ge::graphStatus DoTiling(LITilingInfo *tilingInfo);
    const LITilingData &GetTilingData() const { return tilingData_; }

private:
    ge_helper::TilingContext *context_ = nullptr;
    LITilingData tilingData_;
};

}  // namespace sglang::MlpLIHost

#endif  // MLP_LIGHTNING_INDEXER_TILING_H
