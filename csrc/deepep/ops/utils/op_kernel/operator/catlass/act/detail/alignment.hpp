#ifndef ACT_ALIGNMENT_HPP
#define ACT_ALIGNMENT_HPP

#include "../../act/detail/macros.hpp"

template <uint32_t ALIGN, typename T>
ACT_HOST_DEVICE constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T>
ACT_HOST_DEVICE constexpr T RoundUp(const T &val, const T align)
{
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
ACT_HOST_DEVICE constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
ACT_HOST_DEVICE constexpr T RoundDown(const T val, const T align)
{
    return val / align * align;
}

template <uint32_t DIVISOP, typename T>
ACT_HOST_DEVICE constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOP != 0, "DIVISOP must not be 0");
    return (dividend + DIVISOP - 1) / DIVISOP;
}

template <class T>
ACT_HOST_DEVICE constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

#endif  // ACT_ALIGNMENT_HPP
