#ifndef TLA_NUMERIC_MATH_HPP
#define TLA_NUMERIC_MATH_HPP

#include "../../tla/type_traits.hpp"

namespace tla {

//
// Common Operations
//

template <class T, class U, __TLA_REQUIRES(std::is_arithmetic<T>::value &&std::is_arithmetic<U>::value)>
ACT_HOST_DEVICE constexpr auto max(T const &t, U const &u)
{
    return t < u ? u : t;
}

template <class T, class U, __TLA_REQUIRES(std::is_arithmetic<T>::value &&std::is_arithmetic<U>::value)>
ACT_HOST_DEVICE constexpr auto min(T const &t, U const &u)
{
    return t < u ? t : u;
}

}  // namespace tla

#endif  // TLA_NUMERIC_MATH_HPP
