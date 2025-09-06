#ifndef HOST_MATH_UTILS_H
#define HOST_MATH_UTILS_H
#include <cstdint>

namespace host_utils{
template <typename T>
inline T CeilDiv(const T dividend, const T divisor)
{
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
inline T RoundUp(const T val, const T align = 16)
{
    if (align == 0 || val + align - 1 < val) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline T RoundDown(const T val, const T align = 16)
{
    if (align == 0) {
        return 0;
    }
    return val / align * align;
}
}
#endif
