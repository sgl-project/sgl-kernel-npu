#ifndef ASSIGN_TILING_DATA_H
#define ASSIGN_TILING_DATA_H
#include <assert.h>
#include <cstring>

namespace custom_assign {
#pragma pack(push, 1)
struct CustomAssignTilingData {
    uint32_t batchSize;
    uint32_t tokenPoolLength;
    uint32_t typeBytes;
    uint32_t syncWorkspaceSize;
    uint32_t ubSize;

    void SetToBuffer(uint8_t *dataPtr, size_t dataLen)
    {
        if (dataPtr == nullptr || dataLen < sizeof(CustomAssignTilingData)) {
            return;
        }
        // Ensure no padding is added by the compiler.
        static_assert(sizeof(CustomAssignTilingData) == 5 * sizeof(uint32_t), "CustomAssignTilingData must be packed.");
        memcpy(dataPtr, this, sizeof(CustomAssignTilingData));
    }
};
#pragma pack(pop)
}  // namespace custom_assign
#endif
