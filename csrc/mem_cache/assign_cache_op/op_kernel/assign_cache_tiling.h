#pragma once
#include <cstdint>
namespace optiling {
struct CustomAssignTilingData {
    uint32_t batchSize;
    uint32_t tokenPoolLength;
    uint32_t typeBytes;
    uint32_t syncWorkspaceSize;
    uint32_t ubSize;
};
static_assert(sizeof(CustomAssignTilingData) == 20, "Assign tiling layout must be 20 bytes");
}  // namespace optiling
