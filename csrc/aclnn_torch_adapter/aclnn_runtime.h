#pragma once

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <system_error>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace sgl_cann {

// Resolve from this extension, never from a process-wide libcust_opapi name.
inline void *op_api_symbol(const char *name)
{
    static void *library = [] {
        Dl_info info{};
        TORCH_CHECK(dladdr(reinterpret_cast<void *>(&op_api_symbol), &info) && info.dli_fname,
                    "Cannot locate the CANN Torch extension");
        auto path = std::filesystem::canonical(info.dli_fname).parent_path().parent_path() /
                    "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so";
        auto handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(handle, "Cannot load ", path.string(), ": ", dlerror());
        return handle;  // Keep the library loaded while queued calls can use it.
    }();
    auto symbol = dlsym(library, name);
    TORCH_CHECK(symbol, "Missing CANN symbol: ", name);
    return symbol;
}

// HOST bridges for operators that ship with the CANN toolkit have no symbol in
// the wheel vendor library. Prefer the wheel vendor, then fall back to the
// toolkit's libopapi.so; return nullptr when neither exports the name.
inline void *try_op_api_symbol(const char *name)
{
    static void *vendor = []() -> void * {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void *>(&try_op_api_symbol), &info) || !info.dli_fname) {
            return nullptr;
        }
        std::error_code error;
        const auto extension = std::filesystem::canonical(info.dli_fname, error);
        if (error) {
            return nullptr;
        }
        const auto path =
            extension.parent_path().parent_path() / "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so";
        return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    }();
    if (vendor != nullptr) {
        if (void *symbol = dlsym(vendor, name)) {
            return symbol;
        }
    }
    // Keep the toolkit load lazy, matching the legacy built-in operator bridge.
    static void *builtin = dlopen("libopapi.so", RTLD_LAZY);
    return builtin == nullptr ? nullptr : dlsym(builtin, name);
}

inline void *op_api_symbol_with_fallback(const char *name)
{
    auto symbol = try_op_api_symbol(name);
    TORCH_CHECK(symbol, "Missing CANN symbol: ", name);
    return symbol;
}

inline std::shared_ptr<aclTensor> tensor_descriptor(const at::Tensor &tensor, aclDataType dtype)
{
    // The caller prepares contiguous work tensors. Describe their logical span,
    // including views with an offset, exactly as in the standalone ACLNN test.
    auto descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), 0,
                                      ACL_FORMAT_ND, tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

}  // namespace sgl_cann
