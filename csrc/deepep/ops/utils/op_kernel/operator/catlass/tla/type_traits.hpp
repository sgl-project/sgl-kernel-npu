#ifndef TLA_UTIL_TYPE_TRAITS_HPP
#define TLA_UTIL_TYPE_TRAITS_HPP

#undef inline
#include <tuple>
#define inline __inline__ __attribute__((always_inline))

#define __TLA_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type * = nullptr

namespace tla {

// using std::remove_cvref;
template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// using std::remove_cvref_t;
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

// tuple_size, tuple_element
template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T, std::void_t<typename std::tuple_size<T>::type>>
    : std::integral_constant<size_t, std::tuple_size<T>::value> {};

template <class T>
constexpr size_t tuple_size_v = tuple_size<T>::value;

}  // end namespace tla

#endif  // TLA_UTIL_TYPE_TRAITS_HPP
