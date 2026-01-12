// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ZCCL_H
#define ZCCL_H

#include "acl/acl.h"
#include "cstdint"

enum ZCCLDataType {
    ZCCL_DATA_TYPE_INT8 = 0,
    ZCCL_DATA_TYPE_INT16 = 1,
    ZCCL_DATA_TYPE_INT32 = 2,
    ZCCL_DATA_TYPE_FP16 = 3,
    ZCCL_DATA_TYPE_FP32 = 4,
    ZCCL_DATA_TYPE_INT64 = 5,
    ZCCL_DATA_TYPE_UINT64 = 6,
    ZCCL_DATA_TYPE_UINT8 = 7,
    ZCCL_DATA_TYPE_UINT16 = 8,
    ZCCL_DATA_TYPE_UINT32 = 9,
    ZCCL_DATA_TYPE_FP64 = 10,
    ZCCL_DATA_TYPE_BFP16 = 11,
    ZCCL_DATA_TYPE_RESERVED
};

namespace sglang {
namespace zccl {

inline size_t getSizeFromTypeEnum(ZCCLDataType dtype)
{
    switch (dtype) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        default:
            break;
    }
}

extern "C" int ZcclAllGather(void *input, void *output, uint64_t numel, ZCCLDataType data_type, int team_id, aclrtStream stream);
extern "C" int ZcclAllGatherZeroBuff(void *input, void *output, uint64_t numel, ZCCLDataType data_type, int team_id, aclrtStream stream);

extern "C" int ZcclReduceScatter(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp = 0);

extern "C" int ZcclReduceScatterZeroBuff(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp = 0);

extern "C" int ZcclAllReduce(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp);

}  // namespace zccl
}  // namespace sglang

#endif  // ZCCL_H