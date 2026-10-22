#include <cstdint>
#include <type_traits>

#include SGL_BITMASK_ACLNN_HEADER

using WorkspaceQuery = aclnnStatus (*)(aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
using Execute = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

static_assert(std::is_same_v<decltype(&aclnnSglApplyTokenBitMaskGetWorkspaceSize), WorkspaceQuery>,
              "Expected mutable logits, read-only bitmask, and no separate output tensor");
static_assert(std::is_same_v<decltype(&aclnnSglApplyTokenBitMask), Execute>,
              "Unexpected ACLNN execution signature");
