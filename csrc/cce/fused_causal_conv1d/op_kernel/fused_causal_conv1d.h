#pragma once
#include <acl/acl.h>
#include <stdint.h>
extern "C" aclError cce_launch_causal_conv1d(const uint16_t *, const uint16_t *, uint16_t *, uint16_t *,
                                             const uint16_t *, const int64_t *, const int64_t *, const int64_t *, int,
                                             int, int, int, int, int, int, int, int, aclrtStream);
