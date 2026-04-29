#include "NPUCommon.h"
#include <sstream>
#include <string>

bool OptionsManager::IsHcclZeroCopyEnable = false;
bool OptionsManager::CheckForceUncached = false;

std::string formatErrorCode(int32_t errorCode)
{
    // if (c10_npu::option::OptionsManager::IsCompactErrorOutput()) {
    //     return "";
    // }
    std::ostringstream oss;
    // int deviceIndex = -1;
    // c10_npu::GetDevice(&deviceIndex);
    // auto rank_id = c10_npu::option::OptionsManager::GetRankId();
    oss << "\n[ERROR] CODE" << static_cast<int>(errorCode);

    return oss.str();
}
