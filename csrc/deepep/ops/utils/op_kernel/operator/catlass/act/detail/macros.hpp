#ifndef ACT_DETAIL_MACROS_HPP
#define ACT_DETAIL_MACROS_HPP

#define ACT_DEVICE __forceinline__[aicore]
#define ACT_HOST_DEVICE __forceinline__[host, aicore]
#define ACT_GLOBAL __global__[aicore]

#endif  // ACT_DETAIL_MACROS_HPP
