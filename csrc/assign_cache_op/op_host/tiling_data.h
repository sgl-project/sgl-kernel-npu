#ifndef ASSIGN_TILING_DATA_H
#define ASSIGN_TILING_DATA_H
#include <cstdint>

namespace custom_assign {
struct CustomAssignTilingData {
    // should be aligned
    uint32_t batchSize;
    uint32_t tokenPoolLength;
    uint32_t typeBytes;
    uint32_t syncWorkspaceSize;

    void SetToBuffer(uint8_t *dataPtr, size_t dataLen)
    {
        if (dataPtr == NULL) {
            return;
        }
        size_t offset = 0;
        if (offset + sizeof(uint32_t) > dataLen) {
            return;
        }
        *(uint32_t *)(dataPtr + offset) = batchSize;
        offset += sizeof(uint32_t);

        if (offset + sizeof(uint32_t) > dataLen) {
            return;
        }
        *(uint32_t *)(dataPtr + offset) = tokenPoolLength;
        offset += sizeof(uint32_t);

        if (offset + sizeof(uint32_t) > dataLen) {
            return;
        }
        *(uint32_t *)(dataPtr + offset) = typeBytes;
        offset += sizeof(uint32_t);

        if (offset + sizeof(uint32_t) > dataLen) {
            return;
        }
        *(uint32_t *)(dataPtr + offset) = syncWorkspaceSize;
    }
};
}
#endif
