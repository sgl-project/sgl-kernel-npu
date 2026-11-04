#include "aclnn_runtime.h"
#include "aclnn_sgl_swiglu_group_quant.h"
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <array>

namespace {
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t STATIC_QUANT = 1;
constexpr int64_t MX_QUANT = 2;
constexpr int64_t FP8_QUANT = 3;
constexpr int64_t ACTIVATE_SPLIT = 2;
constexpr int64_t FP8_DIM_ALIGN = 256;
constexpr int64_t FP8_MAX_SCALE_COL_DIVISOR = 128;

//! Allocates y / scale / y_origin with the shape and dtype each quant mode calls for.
std::tuple<at::Tensor, at::Tensor, at::Tensor> ConstructOutputs(const at::Tensor &x, at::ScalarType dst_type,
                                                                int64_t quant_mode, bool ue8m0_scale)
{
    constexpr int64_t SWIGLU_FACTOR = 2;
    constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;

    std::vector<int64_t> y_size(x.sizes().begin(), x.sizes().end());
    int64_t y_last_dim = y_size.back() / SWIGLU_FACTOR;
    y_size.back() = y_last_dim;

    const auto y_dtype = dst_type == at::kFloat8_e5m2 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));

    std::vector<int64_t> scale_size(y_size.begin(), y_size.end());
    if (quant_mode == MX_QUANT) {
        int64_t scale_last_dim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_last_dim = (scale_last_dim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_size.back() = scale_last_dim;
        scale_size.push_back(MX_SCALE_ALIGN_FACTOR);
    } else {
        scale_size.back() = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
    }

    auto scale_type = at::kFloat;
    if (quant_mode == MX_QUANT || (quant_mode == FP8_QUANT && ue8m0_scale)) {
        scale_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale = at::empty(scale_size, x.options().dtype(scale_type));
    at::Tensor y_origin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::make_tuple(y, scale, y_origin);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> swiglu_group_quant(
    const at::Tensor &x, const c10::optional<at::Tensor> &topk_weight, const c10::optional<at::Tensor> &group_index,
    c10::optional<at::ScalarType> dst_type, int64_t quant_mode, int64_t group_size, bool round_scale, bool ue8m0_scale,
    bool output_origin, int64_t group_list_type, double clamp_value)
{
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension, but got ", x.dim());
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x should be FLOAT16 or BFLOAT16, but got ", x.scalar_type());
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(quant_mode == STATIC_QUANT || quant_mode == MX_QUANT || quant_mode == FP8_QUANT,
                "Unsupported quant mode, only support ", STATIC_QUANT, " (group), ", MX_QUANT, " (mx) or ", FP8_QUANT,
                " (fp8), but got ", quant_mode);

    const at::ScalarType out_dtype = dst_type.value_or(at::kFloat8_e4m3fn);
    TORCH_CHECK(out_dtype == at::kFloat8_e4m3fn || out_dtype == at::kFloat8_e5m2,
                "dst_type must be Float8_e4m3fn or Float8_e5m2, but got ", out_dtype);

    const int64_t d = x.size(-1);
    if (quant_mode == STATIC_QUANT || quant_mode == FP8_QUANT) {
        TORCH_CHECK(d % FP8_DIM_ALIGN == 0, "In group/fp8 quant, the last dim of x should be divisible by ",
                    FP8_DIM_ALIGN, ", actual ", d, ".");
    } else {
        TORCH_CHECK(d % FP8_MAX_SCALE_COL_DIVISOR == 0, "In mx quant, the last dim of x should be divisible by ",
                    FP8_MAX_SCALE_COL_DIVISOR, ", actual ", d, ".");
    }
    TORCH_CHECK(d % ACTIVATE_SPLIT == 0, "In swiglu, the last dim of x should be divisible by ", ACTIVATE_SPLIT,
                ", actual ", d, ".");

    // Optional inputs are detected by the kernel as a null GM pointer (see the `!= nullptr` tests in
    // swiglu_group_quant_perf.h Init), so an absent optional must be passed as a real nullptr rather
    // than an empty tensor's data_ptr. The host's tiling must agree with the kernel on presence, so
    // both sides use "present and non-empty" here.
    const bool has_topk_weight = topk_weight.has_value() && topk_weight->defined() && topk_weight->numel() > 0;
    const bool has_group_index = group_index.has_value() && group_index->defined() && group_index->numel() > 0;

    const c10::DeviceGuard guard(x.device());
    auto outputs = ConstructOutputs(x, out_dtype, quant_mode, ue8m0_scale);
    auto &[y, scale, origin] = outputs;
    int64_t batch_size = 1;
    for (int64_t i = 0; i + 1 < x.dim(); ++i)
        batch_size *= x.size(i);
    if (batch_size == 0) return outputs;
    TORCH_CHECK(group_list_type == 0, "swiglu_group_quant: tiling failed for group_list_type");
    const auto xd = x.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    const bool e5 = out_dtype == at::kFloat8_e5m2;
    std::array<at::Tensor, 6> tensors{
        x,     has_topk_weight ? *topk_weight : at::Tensor{}, has_group_index ? *group_index : at::Tensor{}, y, scale,
        origin};
    std::array<aclDataType, 6> types{xd,
                                     ACL_FLOAT,
                                     ACL_INT64,
                                     e5 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN,
                                     scale.scalar_type() == at::kFloat ? ACL_FLOAT : ACL_FLOAT8_E8M0,
                                     xd};
    std::array<std::shared_ptr<aclTensor>, 6> desc;
    auto stream = c10_npu::getCurrentNPUStream();
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].defined()) {
            desc[i] = sgl_cann::tensor_descriptor(tensors[i], types[i]);
            tensors[i].record_stream(stream);
        }
    }
    static auto query = reinterpret_cast<decltype(&aclnnSglSwigluGroupQuantGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSwigluGroupQuantGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglSwigluGroupQuant)>(sgl_cann::op_api_symbol("aclnnSglSwigluGroupQuant"));
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(desc[0].get(), desc[1].get(), desc[2].get(), quant_mode, round_scale, ue8m0_scale,
                        output_origin, group_list_type, clamp_value, e5, clamp_value != 0.0, desc[3].get(),
                        desc[4].get(), desc[5].get(), &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "SwiGLU workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= INT64_MAX, "SwiGLU workspace exceeds tensor allocation limit");
    auto ws = at::empty({static_cast<int64_t>(bytes)}, x.options().dtype(at::kByte));
    ws.record_stream(stream);
    auto launch = [ws, bytes, executor, tensors, desc, stream]() -> int {
        auto result = execute(bytes ? ws.data_ptr() : nullptr, bytes, executor, stream.stream(false));
        TORCH_CHECK(result == ACL_SUCCESS, "SwiGLU execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSwigluGroupQuant", launch);
    return outputs;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "swiglu_group_quant(Tensor x, Tensor? topk_weight=None, Tensor? group_index=None, "
        "ScalarType? dst_type=None, int quant_mode=1, int group_size=128, bool round_scale=False, "
        "bool ue8m0_scale=False, bool output_origin=False, int group_list_type=0, "
        "float clamp_value=0.0) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("swiglu_group_quant", &swiglu_group_quant);
}
