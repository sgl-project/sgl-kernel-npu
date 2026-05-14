#ifndef __SOC_VERSION_COMPAT_H__
#define __SOC_VERSION_COMPAT_H__

#include <string>
#include "tiling/platform/platform_ascendc.h"

#if defined(USE_CANN83_PATH) || defined(USE_CANN82_PATH)
#include "platform/platform_infos_def.h"
#endif

namespace Cam {

inline bool CheckIsA2Chip(const std::string &socVersion)
{
    return socVersion == "Ascend910B" || socVersion == "Ascend910B1";
}

inline std::string GetSocVersionCompat()
{
#if defined(USE_CANN83_PATH) || defined(USE_CANN82_PATH)
    // For CANN 8.2/8.3, use PlatFormInfos (需要从context获取，这里作为fallback)
    return "Unknown";
#else
    // For CANN 8.5+, use platform_ascendc API
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    platform_ascendc::SocVersion socVersionEnum = ascendcPlatform->GetSocVersion();
    
    switch (socVersionEnum) {
        case platform_ascendc::SocVersion::ASCEND910B:
            return "Ascend910B";
        case platform_ascendc::SocVersion::ASCEND910_9382:
            return "Ascend910_9382";
        default:
            return "Unknown";
    }
#endif
}

#if defined(USE_CANN83_PATH) || defined(USE_CANN82_PATH)
inline std::string GetSocVersionFromContext(gert::TilingContext &context)
{
    fe::PlatFormInfos *platformInfoPtr = context.GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return "Unknown";
    }
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    return socVersion;
}
#endif

inline bool CheckIsA2ChipCompat()
{
    std::string socVersion = GetSocVersionCompat();
    return CheckIsA2Chip(socVersion);
}

} // namespace Cam

#endif // __SOC_VERSION_COMPAT_H__