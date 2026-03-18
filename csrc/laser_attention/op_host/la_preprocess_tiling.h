/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef __SRC_OPS_HOST_LA_PREPROCESS_H__
#define __SRC_OPS_HOST_LA_PREPROCESS_H__

#include <register/tilingdata_base.h>


namespace optiling {


struct LaPreprocessCompileInfo {
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t l1Size;
    uint64_t ubSize;
    uint32_t maxAicCoresNum;
    uint32_t maxAivCoresNum;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};

BEGIN_TILING_DATA_DEF(LaPreprocessTiling)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, qSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, kSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, vSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, headNum);
    TILING_DATA_FIELD_DEF(uint32_t, alignLen);
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LaPreprocess, LaPreprocessTiling)


} // namespace optiling

#endif // __SRC_OPS_HOST_LA_PREPROCESS_H__
