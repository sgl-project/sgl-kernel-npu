#include "cache_loc_torch_adapter.h"
#include "aclnn_sgl_cache_loc_assign.h"
#include <torch/library.h>

namespace {
at::Tensor cache_loc_assign(at::Tensor req_indices, at::Tensor token_pool, at::Tensor start_offset,
                            at::Tensor end_offset, at::Tensor out_cache_loc)
{
    static auto query = reinterpret_cast<decltype(&aclnnSglCacheLocAssignGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglCacheLocAssignGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglCacheLocAssign)>(sgl_cann::op_api_symbol("aclnnSglCacheLocAssign"));
    return sgl_cache_loc::Launch(req_indices, token_pool, start_offset, end_offset, out_cache_loc, false,
                                 "aclnnSglCacheLocAssign", query, execute);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "cache_loc_assign(Tensor req_indices, Tensor(a!) token_pool, "
        "Tensor start_offset, Tensor end_offset, Tensor out_cache_loc) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("cache_loc_assign", &cache_loc_assign);
}
