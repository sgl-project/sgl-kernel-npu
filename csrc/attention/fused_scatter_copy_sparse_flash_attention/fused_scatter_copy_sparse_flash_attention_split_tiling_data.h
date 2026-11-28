#pragma once

#include "fused_copy_sfa_tiling_data.h"

struct alignas(8) CannCopySfaTiling {
    FusedCopySfaTilingData compute;
    uint64_t dramCkvAddress;
    uint64_t dramRopeAddress;
    uint32_t combinedHbm;
};
static_assert(offsetof(CannCopySfaTiling, compute) == 0);
