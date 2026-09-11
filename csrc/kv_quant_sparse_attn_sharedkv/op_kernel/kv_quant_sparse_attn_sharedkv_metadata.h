









/*!
 * \file kv_quant_sparse_attn_sharedkv_metadata.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t SAS_META_SIZE = 1024;
using SAS_METADATA_T = int32_t;

constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV=false)
{
    if (isAIV) {
        return FA_METADATA_SIZE * AIC_CORE_NUM + FD_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
    struct SasMetaData {
        uint32_t faMetadata[AIC_CORE_NUM][FA_METADATA_SIZE];
        uint32_t fdMetadata[AIV_CORE_NUM][FD_METADATA_SIZE];
    };
};

static_assert(SAS_META_SIZE * sizeof(SAS_METADATA_T) >= sizeof(detail::SasMetaData));
};

#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_H
