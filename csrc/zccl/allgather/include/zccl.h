#include "acl/acl.h"

enum class ZCCLDataType {
    ZCCL_DATA_TYPE_INT8 = 0;
    ZCCL_DATA_TYPE_INT16 = 1;
    ZCCL_DATA_TYPE_INT32 = 2;
    ZCCL_DATA_TYPE_FP16 = 3;
    ZCCL_DATA_TYPE_FP32 = 4;
    ZCCL_DATA_TYPE_INT64 = 5;
    ZCCL_DATA_TYPE_BFP16 = 6;
};

namespace sglang {
namespace zccl {

extern "C" int zccl_all_gather(void *input, void *output, uint64_t numel, ZCCLDataType data_type, int team_id, aclrtStream stream);

}  // namespace zccl
}  // namespace sglang
