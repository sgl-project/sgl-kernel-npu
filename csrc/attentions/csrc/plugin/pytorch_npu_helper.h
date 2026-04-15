/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)
#ifndef PYTORCH_NPU_HELPER_H
#define PYTORCH_NPU_HELPER_H

#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/aclnn_base.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <fstream>
#include <string>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "find_op_path.h"

#define NPU_NAME_SPACE at_npu::native

using AclOpExecutor = struct aclOpExecutor;
using AclTensor = struct aclTensor;
using AclScalar = struct aclScalar;
using AclIntArray = struct aclIntArray;
using AclFloatArray = struct aclFloatArray;
using AclBoolArray = struct aclBoolArray;
using AclTensorList = struct aclTensorList;

template<typename T = void>
using FunctionPtr = T*;
constexpr int K_HASH_BUF_SIZE = 8192;
constexpr int K_HASH_BUF_MAX_SIZE = K_HASH_BUF_SIZE + 1024;
constexpr int64_t ACL_TENSOR_MAX_DIM_FOR_FORMAT = 5;
constexpr int64_t DIM_NUM_3D = 3;
constexpr int64_t DIM_NUM_4D = 4;
constexpr int64_t DIM_NUM_5D = 5;
extern thread_local char g_hashBuf[K_HASH_BUF_SIZE];
extern thread_local int g_hashOffset;

template <std::string_view const& ApiName>
inline std::string GetWorkspaceSizeApiName()
{
    constexpr std::string_view suffix = "GetWorkspaceSize";
    std::string result(ApiName);
    result += suffix;
    return result;
}

constexpr aclDataType K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
    ACL_UINT8,
    ACL_INT8,
    ACL_INT16,
    ACL_INT32,
    ACL_INT64,
    ACL_FLOAT16,
    ACL_FLOAT,
    ACL_DOUBLE,
    ACL_DT_UNDEFINED,
    ACL_COMPLEX64,
    ACL_COMPLEX128,
    ACL_BOOL,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_BF16,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED,
    ACL_DT_UNDEFINED
};

template<typename T>
inline bool CheckDataPointer(const T* data)
{
    if (data == nullptr) {
        TORCH_CHECK(false, "memcpy failed: source data is null pointer");
        return false;
    }
    return true;
}
inline bool CheckDataSize(size_t size)
{
    if (size == 0) {
        TORCH_CHECK(false, "memcpy failed: copy size is 0 (no data to copy)");
        return false;
    }
    return true;
}
inline bool CheckBufferSpace(size_t size)
{
    if (g_hashOffset + size > K_HASH_BUF_SIZE) {
        g_hashOffset = K_HASH_BUF_MAX_SIZE;
        TORCH_CHECK(false, "memcpy failed: buffer overflow");
        return false;
    }
    return true;
}
template<typename T>
inline bool ValidateMemcpyParams(const T* data, size_t size)
{
    return CheckDataPointer(data) && CheckDataSize(size) && CheckBufferSpace(size);
}


inline bool IsCustomLibPathEmpty()
{
    return g_customLibPath.empty();
}


inline bool ShouldSearchCustomLib()
{
    return !IsCustomLibPathEmpty();
}
inline void* SearchCustomLibPaths(const char* apiName)
{
    for (const auto &libPath : g_customLibPath) {
        void* funcAddr = FindFuncInCustomLibPath(apiName, libPath);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return nullptr;
}

inline void LogCustomLibNotFound(const char* apiName)
{
    ASCEND_LOGI("%s is not in custom lib.", apiName);
}
inline void* FindFuncInCustomLib(const char* apiName)
{
    if (!ShouldSearchCustomLib()) {
        return nullptr;
    }
    
    void* result = SearchCustomLibPaths(apiName);
    if (result == nullptr) {
        LogCustomLibNotFound(apiName);
    }
    return result;
}
inline bool IsDefaultCustomLibPathEmpty()
{
    return g_defaultCustomLibPath.empty();
}
inline bool ShouldSearchDefaultLib()
{
    return !IsDefaultCustomLibPathEmpty();
}
inline void* SearchDefaultLibPaths(const char* apiName)
{
    for (const auto &libPath : g_defaultCustomLibPath) {
        void* funcAddr = FindFuncInDefaultLibPath(apiName, libPath);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return nullptr;
}
inline void LogDefaultLibNotFound(const char* apiName)
{
    ASCEND_LOGI("%s is not in default custom lib.", apiName);
}
inline void* FindFuncInDefaultLib(const char* apiName)
{
    if (!ShouldSearchDefaultLib()) {
        return nullptr;
    }
    void* result = SearchDefaultLibPaths(apiName);
    if (result == nullptr) {
        LogDefaultLibNotFound(apiName);
    }
    return result;
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    void* funcAddr = FindFuncInCustomLib(apiName);
    if (funcAddr != nullptr) {
        return funcAddr;
    }
    funcAddr = FindFuncInDefaultLib(apiName);
    if (funcAddr != nullptr) {
        return funcAddr;
    }
    return GetFuncFromDefaultLib(apiName);
}
c10::Scalar CreateScalarFromDouble(const at::Tensor* tensor);
c10::Scalar CreateScalarFromLong(const at::Tensor* tensor);
c10::Scalar CreateScalarFromFloat(const at::Tensor* tensor);
c10::Scalar CreateScalarFromInt(const at::Tensor* tensor);
c10::Scalar CreateScalarFromHalf(const at::Tensor* tensor);
c10::Scalar CreateScalarFromBool(const at::Tensor* tensor);
c10::Scalar CreateScalarFromComplexDouble(const at::Tensor* tensor);
c10::Scalar CreateScalarFromComplexFloat(const at::Tensor* tensor);
c10::Scalar CreateScalarFromBFloat16(const at::Tensor* tensor);
inline c10::Scalar CreateScalarFromDouble(const at::Tensor* aclInput)
{
    double value = *(double *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromLong(const at::Tensor* aclInput)
{
    int64_t value = *(int64_t *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromFloat(const at::Tensor* aclInput)
{
    float value = *(float *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromInt(const at::Tensor* aclInput)
{
    int value = *(int *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromHalf(const at::Tensor* aclInput)
{
    c10::Half value = *(c10::Half *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromBool(const at::Tensor* aclInput)
{
    int8_t value = *(int8_t *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromComplexDouble(const at::Tensor* aclInput)
{
    c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromComplexFloat(const at::Tensor* aclInput)
{
    c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
    return c10::Scalar(value);
}

inline c10::Scalar CreateScalarFromBFloat16(const at::Tensor* aclInput)
{
    c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
    return c10::Scalar(value);
}
inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
    const at::Tensor *aclInput = &tensor;
    switch (aclInput->scalar_type()) {
        case at::ScalarType::Double:
            return CreateScalarFromDouble(aclInput);
        case at::ScalarType::Long:
            return CreateScalarFromLong(aclInput);
        case at::ScalarType::Float:
            return CreateScalarFromFloat(aclInput);
        case at::ScalarType::Int:
            return CreateScalarFromInt(aclInput);
        case at::ScalarType::Half:
            return CreateScalarFromHalf(aclInput);
        case at::ScalarType::Bool:
            return CreateScalarFromBool(aclInput);
        case at::ScalarType::ComplexDouble:
            return CreateScalarFromComplexDouble(aclInput);
        case at::ScalarType::ComplexFloat:
            return CreateScalarFromComplexFloat(aclInput);
        case at::ScalarType::BFloat16:
            return CreateScalarFromBFloat16(aclInput);
        default:
            return c10::Scalar();
    }
}
inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpuTensor)
{
    at::Tensor cpuPinMemTensor = cpuTensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex),
                              cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpuScalar, at::ScalarType scalarDataType)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpuScalar).to(scalarDataType));
}

inline AclTensor *ConvertType(const at::Tensor &atTensor)
{
    if (!atTensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalarDataType = atTensor.scalar_type();
    aclDataType aclType = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarDataType)];
    TORCH_CHECK(aclType != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalarDataType)) + " has not been supported")
    c10::SmallVector<int64_t, ACL_TENSOR_MAX_DIM_FOR_FORMAT> storageDims;
    // if aclType is ACL_STRING, storageDims is empty.
    auto itemSize = atTensor.itemsize();
    if (itemSize == 0) {
        AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
        return nullptr;
    }
    if (aclType != ACL_STRING) {
        storageDims.push_back(atTensor.storage().nbytes() / itemSize);
    }

    const auto dimNum = atTensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case DIM_NUM_3D:
            format = ACL_FORMAT_NCL;
            break;
        case DIM_NUM_4D:
            format = ACL_FORMAT_NCHW;
            break;
        case DIM_NUM_5D:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }

    if (atTensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar expScalar = ConvertTensorToScalar(atTensor);
        at::Tensor aclInput = CopyScalarToDevice(expScalar, scalarDataType);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), aclType,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto aclTensorObj =
        aclCreateTensor(atTensor.sizes().data(), atTensor.sizes().size(), aclType, atTensor.strides().data(),
                        atTensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(atTensor.storage().data()));
    return aclTensorObj;
}

inline AclScalar *ConvertType(const at::Scalar &atScalar)
{
    at::ScalarType scalarDataType = atScalar.type();
    aclDataType aclType = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarDataType)];
    TORCH_CHECK(aclType != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalarDataType)) + " has not been supported")
    AclScalar *aclScalarObj = nullptr;
    switch (scalarDataType) {
        case at::ScalarType::Double: {
            double value = atScalar.toDouble();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = atScalar.toLong();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = atScalar.toBool();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = atScalar.toComplexDouble();
            aclScalarObj = aclCreateScalar(&value, aclType);
            break;
        }
        default:
            aclScalarObj = nullptr;
            break;
    }
    return aclScalarObj;
}

inline AclIntArray *ConvertType(const at::IntArrayRef &atArray)
{
    auto array = aclCreateIntArray(atArray.data(), atArray.size());
    return array;
}

template <std::size_t N> inline AclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline AclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline AclTensorList *ConvertType(const at::TensorList &atTensorList)
{
    std::vector<const AclTensor *> tensorTist(atTensorList.size());
    for (size_t i = 0; i < atTensorList.size(); i++) {
        tensorTist[i] = ConvertType(atTensorList[i]);
    }
    auto aclTensorList = aclCreateTensorList(tensorTist.data(), tensorTist.size());
    return aclTensorList;
}

inline AclTensor *ConvertType(const c10::optional<at::Tensor> &optTensor)
{
    if (optTensor.has_value() && optTensor.value().defined()) {
        return ConvertType(optTensor.value());
    }
    return nullptr;
}

inline AclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &optArray)
{
    if (optArray.has_value()) {
        return ConvertType(optArray.value());
    }
    return nullptr;
}

inline AclScalar *ConvertType(const c10::optional<at::Scalar> &optScalar)
{
    if (optScalar.has_value()) {
        return ConvertType(optScalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarType)];
}

template <typename T> T ConvertType(T value)
{
    return value;
}

template<typename TargetFuncType, typename SourceType>
struct FunctionPointerConverter {
    static TargetFuncType Convert(SourceType ptr)
    {
        static_assert(sizeof(TargetFuncType) == sizeof(SourceType), "Function pointer size mismatch");
        static_assert(std::is_pointer_v<SourceType>, "SourceType must be a pointer type");
        static_assert(std::is_pointer_v<TargetFuncType>, "TargetFuncType must be a function pointer type");
        union {
            SourceType ptr;
            TargetFuncType func;
        } converter;
        converter.ptr = ptr;
        return converter.func;
    }
};

template <typename Tuple, size_t... I, typename FuncPtrType>
auto ConvertToOpApiFunc(const Tuple &params, FuncPtrType *opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = FunctionPointerConverter<OpApiFunc, FuncPtrType*>::Convert(opApiAddr);
    return func;
}

template <typename Tuple, typename FuncPtrType>
auto ConvertToOpApiFunc(const Tuple &params, FuncPtrType *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(AclTensor *p)
{
    aclDestroyTensor(p);
}

inline void Release(AclScalar *p)
{
    aclDestroyScalar(p);
}

inline void Release(AclIntArray *p)
{
    aclDestroyIntArray(p);
}

inline void Release(AclBoolArray *p)
{
    aclDestroyBoolArray(p);
}

inline void Release(AclTensorList *p)
{
    aclDestroyTensorList(p);
}

template <typename T> void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I> void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple> void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I> auto Call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple> auto Call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return Call(f, t, std::make_index_sequence<size>{});
}

uint64_t CalcHashId();
using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = void (*)(void *, bool);
using ReleaseHugeMem = void (*)(void *, bool);

template<typename GetWorkspaceSizeFuncType, typename OpApiFuncType>
inline void ValidateApiAddresses(
    GetWorkspaceSizeFuncType getWorkspaceSizeFuncAddr,
    OpApiFuncType opApiFuncAddr,
    std::string_view apiName,
    std::string_view workspaceSizeApiStr)
{
    TORCH_CHECK(
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        apiName.data(), " or ", workspaceSizeApiStr.data(), " not in ", GetOpApiLibName(),
        ", or ", GetOpApiLibName(), " not found."
    );
}

template<typename InitMemAddrType>
inline void InitHugeMemCustom(InitMemAddrType initMemAddr)
{
    using InitHugeMemFunc = int (*)(FunctionPtr<>, bool);
    auto initMemFunc = FunctionPointerConverter<InitHugeMemFunc, InitMemAddrType>::Convert(initMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }
}

template <std::string_view const& ApiName, typename GetWorkspaceSizeFuncType, typename... Args>
auto PrepareParamsAndCalcWorkspaceSize(
    uint64_t* workspaceSizeAddr, AclOpExecutor** executorAddr,
    GetWorkspaceSizeFuncType getWorkspaceSizeFuncAddr, Args&&... args)
{
    auto convertedParams = ConvertTypes(std::forward<Args>(args)..., workspaceSizeAddr, executorAddr);
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr);
    auto workspaceStatus = Call(getWorkspaceSizeFunc, convertedParams);

    TORCH_CHECK(
        workspaceStatus == 0,
        "call ", ApiName.data(), " failed, detail:", aclGetRecentErrMsg()
    );

    return convertedParams;
}

template <typename ReleaseMemAddrType>
inline void ReleaseHugeMemResource(ReleaseMemAddrType releaseMemAddr)
{
    using ReleaseHugeMemFunc = void (*)(FunctionPtr<>, bool);
    auto releaseMemFunc = FunctionPointerConverter<ReleaseHugeMemFunc, ReleaseMemAddrType>::Convert(releaseMemAddr);
    if (releaseMemFunc) {
        releaseMemFunc(nullptr, false);
    }
}

template <typename UnInitMemAddrType>
inline void UnInitHugeMem(UnInitMemAddrType unInitMemAddr)
{
    using UnInitHugeMemFunc = void (*)(FunctionPtr<>, bool);
    auto unInitMemFunc = FunctionPointerConverter<UnInitHugeMemFunc, UnInitMemAddrType>::Convert(unInitMemAddr);
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
}

template <std::string_view const& ApiName, typename... Args>
void EXEC_NPU_CMD(Args&&... args)
{
    auto workspaceSizeApiStr = GetWorkspaceSizeApiName<ApiName>();
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspaceSizeApiStr.c_str());
    static const auto opApiFuncAddr = GetOpApiFuncAddr(ApiName.data());
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");

    ValidateApiAddresses(
        getWorkspaceSizeFuncAddr,
        opApiFuncAddr,
        ApiName,
        std::string_view(workspaceSizeApiStr.c_str(), workspaceSizeApiStr.length())
    );

    InitHugeMemCustom(initMemAddr);
    uint64_t workspaceSize = 0;
    AclOpExecutor* executor = nullptr;
    auto convertedParams = PrepareParamsAndCalcWorkspaceSize<ApiName>(
        &workspaceSize,
        &executor,
        getWorkspaceSizeFuncAddr,
        std::forward<Args>(args)...
    );
    at::Tensor workspaceTensor;
    void* workspaceAddr = AllocateWorkspace(workspaceSize, workspaceTensor);
    auto aclStreamObj = c10_npu::getCurrentNPUStream().stream(false);
    auto aclCall = [convertedParams, workspaceAddr, workspaceSize,
        aclStreamObj, executor]() -> int {
        using OpApiFunc = int (*)(FunctionPtr<>, uint64_t, AclOpExecutor*, const aclrtStream);
        auto opApiFunc = FunctionPointerConverter<OpApiFunc, void*>::Convert(opApiFuncAddr);
        auto apiRet = opApiFunc(workspaceAddr, workspaceSize, executor, aclStreamObj);
        TORCH_CHECK(
            apiRet == 0,
            "call ", ApiName.data(), " failed, detail:", aclGetRecentErrMsg()
        );
        ReleaseConvertTypes(convertedParams);
        ReleaseHugeMemResource(releaseMemAddr);
        return apiRet;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(ApiName.data());
    cmd.SetCustomHandler(aclCall);
    cmd.Run();
    UnInitHugeMem(unInitMemAddr);
}

#endif // PYTORCH_NPU_HELPER_H