#include "cache_loc_torch_adapter.h"
#include "aclnn_sgl_cache_loc_update.h"
#include <torch/library.h>

namespace {
at::Tensor cache_loc_update(at::Tensor req_indices, at::Tensor token_pool, at::Tensor start_offset,
                            at::Tensor end_offset, at::Tensor out_cache_loc)
{
    static auto query = reinterpret_cast<decltype(&aclnnSglCacheLocUpdateGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglCacheLocUpdateGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglCacheLocUpdate)>(sgl_cann::op_api_symbol("aclnnSglCacheLocUpdate"));
    return sgl_cache_loc::Launch(req_indices, token_pool, start_offset, end_offset, out_cache_loc, true,
                                 "aclnnSglCacheLocUpdate", query, execute);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "cache_loc_update(Tensor req_indices, Tensor token_pool, "
        "Tensor start_offset, Tensor end_offset, Tensor(a!) out_cache_loc) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("cache_loc_update", &cache_loc_update);
}
