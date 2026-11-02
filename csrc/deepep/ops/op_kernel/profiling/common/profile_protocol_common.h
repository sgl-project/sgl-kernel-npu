#ifndef DEEPEP_OP_KERNEL_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_H
#define DEEPEP_OP_KERNEL_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define DEEPEP_PROFILE_INLINE __aicore__ inline
#else
#define DEEPEP_PROFILE_INLINE inline
#endif

#include "../../../profiling/common/profile_protocol_common_core.h"

#undef DEEPEP_PROFILE_INLINE

#endif  // DEEPEP_OP_KERNEL_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_H
