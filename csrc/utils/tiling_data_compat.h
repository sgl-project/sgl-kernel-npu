#pragma once

#include <cstdio>
#include "register/tilingdata_base.h"

// CANN's array setters call TilingDef::GeLogError(const std::string&).
// Some CANN builds export it only with the old libstdc++ string ABI, whereas
// this extension must use PyTorch's ABI (which can be the new ABI). Handle the
// literal diagnostic in the generated class instead of passing std::string
// across that boundary. Keep the SDK's field storage and serialization intact.
// This does not redefine SDK macros or provide a replacement libregister symbol.
#define SGL_BEGIN_TILING_DATA_DEF(class_name)                               \
    BEGIN_TILING_DATA_DEF(class_name)                                       \
private:                                                                    \
    static void GeLogError(const char *message)                             \
    {                                                                       \
        std::fprintf(stderr, "[sgl_kernel_npu tiling] %s\n", message);      \
    }
