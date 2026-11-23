// Host dispatch for the CCE sparse flash attention operator: selects the
// bisheng-compiled kernel variant matching the runtime head count. The four
// variant launchers live in the bisheng objects archived next to this file.
#include <acl/acl.h>

#define DECLARE_VARIANT(H)                                                                       \
    extern "C" aclError cce_launch_sfa_h##H(                                                     \
        void*, void*, void*, void*, void*, void*, void*, void*, int, int, int, aclrtStream)
DECLARE_VARIANT(16);
DECLARE_VARIANT(32);
DECLARE_VARIANT(64);
DECLARE_VARIANT(128);

extern "C" aclError cce_launch_sfa(void* q, void* qr, void* kv, void* kr, void* ids,
    void* out, void* maximum, void* sum, int heads, int queries, int groups, aclrtStream stream) {
    if (queries <= 0 || groups <= 0) return ACL_ERROR_INVALID_PARAM;
    switch (heads) {
#define DISPATCH(H)                                                                              \
    case H:                                                                                      \
        return cce_launch_sfa_h##H(q, qr, kv, kr, ids, out, maximum, sum, heads, queries,        \
            groups, stream)
        DISPATCH(16);
        DISPATCH(32);
        DISPATCH(64);
        DISPATCH(128);
#undef DISPATCH
        default:
            return ACL_ERROR_INVALID_PARAM;
    }
}
