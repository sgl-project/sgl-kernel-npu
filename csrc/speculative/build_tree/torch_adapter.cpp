// Framework bridge for the independent SglBuildTree CANN operator.
// Five Ref arguments retain caller-owned storage; the legacy registration stays intact.
#include "aclnn_runtime.h"
#include "aclnn_sgl_build_tree.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <limits>
#include <vector>

namespace {
std::shared_ptr<aclTensor> raw_pointer_descriptor(const at::Tensor &tensor, aclDataType dtype)
{
    // The old kernel treats data_ptr as dense storage regardless of Tensor strides.
    // Describe that same span, without a contiguous copy or new copyback semantics.
    std::vector<int64_t> strides(tensor.dim());
    int64_t stride = 1;
    for (int64_t dim = tensor.dim(); dim-- > 0;) {
        strides[dim] = stride;
        stride *= tensor.size(dim);
    }
    auto *desc = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                 tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(desc, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {desc, [](aclTensor *value) { aclDestroyTensor(value); }};
}

void build_tree(const at::Tensor &parentList, const at::Tensor &selectedIndex, const at::Tensor &verifiedSeqLen,
                const at::Tensor &treeMask, const at::Tensor &positions, const at::Tensor &retriveIndex,
                const at::Tensor &retriveNextToken, const at::Tensor &retriveNextSibling, int64_t topk, int64_t depth,
                int64_t draftTokenNum, int64_t treeMaskMode)
{
    TORCH_CHECK(treeMaskMode != 2, "QLEN_ONLY_BITPACKING is not implemented");
    TORCH_CHECK(parentList.device().type() == c10::DeviceType::PrivateUse1, "parent_list must be on an NPU");
    const c10::DeviceGuard deviceGuard(parentList.device());
    std::vector<at::Tensor> tensors{parentList, selectedIndex, verifiedSeqLen,   treeMask,
                                    positions,  retriveIndex,  retriveNextToken, retriveNextSibling};
    std::vector<std::shared_ptr<aclTensor>> descriptors;
    for (size_t index = 0; index < tensors.size(); ++index) {
        TORCH_CHECK(tensors[index].device() == parentList.device(), "all build_tree tensors must be on the same NPU");
        TORCH_CHECK(tensors[index].scalar_type() == (index == 3 ? at::kBool : at::kLong),
                    "build_tree requires int64 tensors and a bool tree_mask");
        descriptors.push_back(raw_pointer_descriptor(tensors[index], index == 3 ? ACL_BOOL : ACL_INT64));
    }
    // Generated prototypes check Ref merging, attribute order and argument types.
    static auto query = reinterpret_cast<decltype(&aclnnSglBuildTreeGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglBuildTreeGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglBuildTree)>(sgl_cann::op_api_symbol("aclnnSglBuildTree"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                        descriptors[4].get(), descriptors[5].get(), descriptors[6].get(), descriptors[7].get(), topk,
                        depth, draftTokenNum, treeMaskMode, &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "build_tree workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, parentList.options().dtype(at::kByte));
    tensors.push_back(workspace);
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Captures cover the host queue; stream recording covers asynchronous access.
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "build_tree ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglBuildTree", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "build_tree(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, "
        "Tensor(a!) tree_mask, Tensor(b!) positions, Tensor(c!) retrive_index, "
        "Tensor(d!) retrive_next_token, Tensor(e!) retrive_next_sibling, "
        "int topk, int depth, int draft_token_num, int tree_mask_mode) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("build_tree", &build_tree);
}
