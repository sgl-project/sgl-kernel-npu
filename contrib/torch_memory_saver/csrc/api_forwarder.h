#pragma once
#include <acl/acl.h>
#include <dlfcn.h>
namespace APIForwarder {
aclError call_real_aclrt_malloc_align32(void **ptr, size_t size,
                                        aclrtMemMallocPolicy policy);
aclError call_real_aclrt_free(void *ptr);
} // namespace APIForwarder
