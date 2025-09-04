#pragma once
#include <dlfcn.h>
#include <acl/acl.h>
namespace APIForwarder {
    aclError call_real_aclrt_malloc_align32(void **ptr, size_t size, aclrtMemMallocPolicy policy);
    aclError call_real_aclrt_free(void *ptr);
}
