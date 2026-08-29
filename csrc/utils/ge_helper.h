#ifndef SGLANG_KERNEL_GE_HELPER_H
#define SGLANG_KERNEL_GE_HELPER_H
#include <cstring>
#include <cstdint>
#include <any>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

#include <stdexcept>

constexpr ge::DataType SCALAR_TYPE_TO_GE_DATATYPE(at::ScalarType scalarType)
{
    switch (scalarType) {
        case at::ScalarType::Float:
            return ge::DT_FLOAT;
        case at::ScalarType::Half:
            return ge::DT_FLOAT16;
        case at::ScalarType::Char:
            return ge::DT_INT8;
        case at::ScalarType::Int:
            return ge::DT_INT32;
        case at::ScalarType::Byte:
            return ge::DT_UINT8;
        case at::ScalarType::Short:
            return ge::DT_INT16;
        case at::ScalarType::UInt16:
            return ge::DT_UINT16;
        case at::ScalarType::UInt32:
            return ge::DT_UINT32;
        case at::ScalarType::Long:
            return ge::DT_INT64;
        case at::ScalarType::UInt64:
            return ge::DT_UINT64;
        case at::ScalarType::Double:
            return ge::DT_DOUBLE;
        case at::ScalarType::Bool:
            return ge::DT_BOOL;
        case at::ScalarType::BFloat16:
            return ge::DT_BF16;
        default:
            return ge::DT_FLOAT16;
    }
}

constexpr uint32_t GE_DATATYPE_TO_KEY(ge::DataType geDatatype)
{
    switch (geDatatype) {
        case ge::DT_FLOAT:
            return 0;
        case ge::DT_FLOAT16:
            return 1;
        case ge::DT_INT8:
            return 2;
        case ge::DT_INT32:
            return 3;
        case ge::DT_UINT8:
            return 4;
        case ge::DT_INT16:
            return 5;
        case ge::DT_UINT16:
            return 6;
        case ge::DT_UINT32:
            return 7;
        case ge::DT_INT64:
            return 8;
        case ge::DT_UINT64:
            return 9;
        case ge::DT_DOUBLE:
            return 10;
        case ge::DT_BOOL:
            return 11;
        case ge::DT_BF16:
            return 12;
        default:
            return 1;
    }
}
namespace sglang {
namespace ge_helper {

enum class ParamTypeCls : uint32_t {
    REQUIRED = 0,
    OPTIONAL,
};
using AttrTypeCls = ParamTypeCls;
constexpr auto REQUIRED = ParamTypeCls::REQUIRED;
constexpr auto OPTIONAL = ParamTypeCls::OPTIONAL;
gert::StorageShape CreateStorageShape(const std::vector<int64_t> origin, const std::vector<int64_t> storage);

constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIM3 = 3;
constexpr size_t DIM4 = 4;
constexpr int64_t MAX_TILING_CACHE_ENTRIES = 512;

inline bool IsNpuGraphCapturing()
{
    aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model = nullptr;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto status = aclmdlRICaptureGetInfo(stream, &captureStatus, &model);
    TORCH_CHECK(status == ACL_ERROR_NONE, "[GE_Helper] Failed to query NPU graph capture status, acl error ", status);
    return captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

template <typename T>
class TilingTensorCache
{
    static_assert(std::is_trivially_copyable_v<T>, "TilingData must be trivially copyable");

public:
    static at::Tensor Get(const T &tilingData, const std::string &opName)
    {
        static std::mutex cacheMutex;
        static std::unordered_map<int64_t, DeviceCache> deviceCaches;

        const auto device = c10_npu::getCurrentNPUStream().device();
        const auto tilingSize = static_cast<int64_t>(sizeof(T));
        // TilingData must be deterministically initialized because its complete
        // serialized representation, including padding, is used as the cache key.
        std::string key(tilingSize, '\0');
        std::memcpy(key.data(), &tilingData, sizeof(T));

        std::lock_guard<std::mutex> lock(cacheMutex);
        auto &cache = deviceCaches[static_cast<int64_t>(device.index())];
        auto iter = cache.slots.find(key);
        if (iter != cache.slots.end()) {
            return cache.buffer.narrow(0, iter->second * tilingSize, tilingSize);
        }

        const bool isCapturing = IsNpuGraphCapturing();
        if (cache.nextSlot >= MAX_TILING_CACHE_ENTRIES) {
            TORCH_CHECK(!isCapturing, opName,
                        ": the current tiling configuration is not cached and the 512-entry cache is full; "
                        "NPU graph capture cannot use a one-shot tiling address");
            auto tilingTensor = at::empty({tilingSize}, at::TensorOptions().device(device).dtype(at::kByte));
            CopyTo_(tilingTensor, tilingData, opName);
            return tilingTensor;
        }

        if (!cache.buffer.defined()) {
            TORCH_CHECK(!isCapturing, opName,
                        ": run one eager warmup with the same configuration before NPU graph capture to initialize "
                        "the tiling cache");
            cache.buffer =
                at::empty({tilingSize * MAX_TILING_CACHE_ENTRIES}, at::TensorOptions().device(device).dtype(at::kByte));
        } else {
            TORCH_CHECK(!isCapturing, opName,
                        ": the current tiling configuration is not cached; run one eager warmup with the same tensor "
                        "shapes, dtypes, optional inputs, and attributes before NPU graph capture");
        }

        const int64_t slot = cache.nextSlot;
        auto cachedTiling = cache.buffer.narrow(0, slot * tilingSize, tilingSize);
        CopyTo_(cachedTiling, tilingData, opName);
        cache.slots.emplace(std::move(key), slot);
        cache.nextSlot++;
        return cachedTiling;
    }

private:
    static void CopyTo_(const at::Tensor &destination, const T &tilingData, const std::string &opName)
    {
        auto status = aclrtMemcpy(destination.data_ptr(), sizeof(T), &tilingData, sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(status == ACL_ERROR_NONE, opName, ": failed to copy tiling data, acl error ", status);
    }
    struct DeviceCache {
        at::Tensor buffer;
        std::unordered_map<std::string, int64_t> slots;
        int64_t nextSlot = 0;
    };
};

class InputDef
{
public:
    InputDef &ParamType(ParamTypeCls type)
    {
        paramType_ = type;
        return *this;
    }

    InputDef &DataType(const std::vector<ge::DataType> &types)
    {
        dataTypes_ = types;
        return *this;
    }

    InputDef &DataTypeList(const std::vector<ge::DataType> &types)
    {
        useDataTypeList_ = true;
        dataTypes_ = types;
        return *this;
    }

    InputDef &Format(const std::vector<ge::Format> &formats)
    {
        formats_ = formats;
        return *this;
    }
    InputDef &FormatList(const std::vector<ge::Format> &formats)
    {
        useFormatList_ = true;
        formats_ = formats;
        return *this;
    }

    InputDef &AutoContiguous()
    {
        autoContiguous_ = true;
        return *this;
    }

    const ge::DataType GetDataType(uint32_t index) const
    {
        if (useDataTypeList_) {
            return dataTypes_[0];
        }
        TORCH_CHECK(index < dataTypes_.size(), "[GE_Helper] InputDef::GetDataType index out of range");
        return dataTypes_[index];
    }

    const std::vector<ge::DataType> &GetDataTypes() const
    {
        return dataTypes_;
    }

    const ge::Format GetFormat(uint32_t index) const
    {
        if (useFormatList_) {
            return formats_[0];
        }
        TORCH_CHECK(index < formats_.size(), "[GE_Helper] InputDef::GetFormat index out of range");
        return formats_[index];
    }

private:
    ParamTypeCls paramType_;
    std::vector<ge::DataType> dataTypes_;
    std::vector<ge::Format> formats_;
    bool autoContiguous_ = false;
    bool useFormatList_ = false;
    bool useDataTypeList_ = false;
};

class AttrDef
{
public:
    AttrDef &AttrType(AttrTypeCls type)
    {
        attrType_ = type;
        return *this;
    }

    AttrDef &String(const std::string &value)
    {
        TORCH_CHECK(valueInitialized_ == false,
                    "[GE_Helper] Cannot set default value for an attribute that has already been initialized.");
        anyValue_ = value;
        valueInitialized_ = true;
        isString_ = true;
        return *this;
    }

    AttrDef &Int(int value)
    {
        TORCH_CHECK(valueInitialized_ == false,
                    "[GE_Helper] Cannot set default value for an attribute that has already been initialized.");
        anyValue_ = value;
        valueInitialized_ = true;
        return *this;
    }

    AttrDef &Float(float value)
    {
        TORCH_CHECK(valueInitialized_ == false,
                    "[GE_Helper] Cannot set default value for an attribute that has already been initialized.");
        anyValue_ = value;
        valueInitialized_ = true;
        return *this;
    }

    AttrDef &Bool(bool value)
    {
        TORCH_CHECK(valueInitialized_ == false,
                    "[GE_Helper] Cannot set default value for an attribute that has already been initialized.");
        anyValue_ = value;
        valueInitialized_ = true;
        return *this;
    }

    const std::any GetValue() const
    {
        return anyValue_;
    }

    const std::string GetString() const
    {
        return strValue_;
    }

    bool IsString()
    {
        return isString_;
    }

    void SetAny(std::any value)
    {
        anyValue_ = value;
    }

    void SetStr(std::string str)
    {
        strValue_ = str;
    }

private:
    AttrTypeCls attrType_;  // REQUIRED or OPTIONAL
    std::any anyValue_;     // need C++17
    bool isString_{false};
    std::string strValue_;
    bool valueInitialized_ = false;
};

class RuntimeAttrs
{
public:
    RuntimeAttrs() = default;

    void AddStr(const std::string &value)
    {
        strValues_.push_back(value);
    }

    void AddAny(const std::any &value)
    {
        anyValues_.push_back(value);
    }

    const char *GetStr(const size_t index) const
    {
        return strValues_[index].c_str();
    }

    template <typename T>
    const T *GetAttrPointer(size_t index)
    {
        std::any &anyValue = anyValues_[index];
        TORCH_CHECK(anyValue.type() == typeid(T), "[GE_Helper] Invalid attribute type.");
        return &std::any_cast<const T &>(anyValue);
    }

private:
    std::vector<std::string> strValues_;
    std::vector<std::any> anyValues_;
};

class TilingContext
{
public:
    TilingContext(const std::string &nodeName) : nodeName_(nodeName) {}

    void RegisterTensor(const c10::optional<at::Tensor> &tensor, bool isInput)
    {
        // convert to gert::Tensor and add to inputTensor_
        // get shape and convert to gert::StorageShape, then add to inputShape_
        std::vector<gert::StorageShape> *shapePtr;
        std::vector<std::shared_ptr<gert::Tensor>> *tensorPtr;
        std::vector<std::shared_ptr<gert::CompileTimeTensorDesc>> *descPtr;

        if (isInput) {
            shapePtr = &inputShape_;
            tensorPtr = &inputTensor_;
            descPtr = &inputDesc_;
        } else {
            shapePtr = &outputShape_;
            tensorPtr = &outputTensor_;
            descPtr = &outputDesc_;
        }

        if (!tensor.has_value()) {
            auto storageShape = CreateStorageShape({}, {});
            shapePtr->emplace_back(std::move(storageShape));
            std::shared_ptr<gert::Tensor> nullTensor(nullptr);
            tensorPtr->push_back(nullTensor);
            return;
        }

        auto shape = tensor.value().sizes();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());

        auto storageShape = CreateStorageShape(shapeVec, shapeVec);
        shapePtr->emplace_back(std::move(storageShape));

        // Safety check to avoid underflow
        TORCH_CHECK(!descPtr->empty(), "[GE_Helper] No tensor description available.");

        auto index = tensorPtr->size();
        TORCH_CHECK(index < descPtr->size(), "[GE_Helper] Tensor registration index out of range.");
        // storageFormat == originFormat
        auto geOriginFormat = (*descPtr)[index]->GetOriginFormat();
        auto storageFormat = gert::StorageFormat(geOriginFormat, geOriginFormat, gert::ExpandDimsType());
        auto dataType = (*descPtr)[index]->GetDataType();
        auto geTensor = std::make_shared<gert::Tensor>(shapePtr->back(), storageFormat, dataType);
        tensorPtr->push_back(geTensor);
    }

    const gert::CompileTimeTensorDesc *GetInputDesc(uint32_t index) const
    {
        return inputDesc_[index].get();
    }

    const gert::StorageShape *GetInputShape(uint32_t index) const
    {
        return &inputShape_[index];
    }

    const gert::Tensor *GetInputTensor(uint32_t index) const
    {
        return inputTensor_[index].get();
    }

    const gert::CompileTimeTensorDesc *GetOptionalInputDesc(uint32_t index) const
    {
        return inputTensor_[index] == nullptr ? nullptr : inputDesc_[index].get();
    }

    const gert::StorageShape *GetOptionalInputShape(uint32_t index) const
    {
        return &inputShape_[index];
    }

    const gert::Tensor *GetOptionalInputTensor(uint32_t index) const
    {
        return inputTensor_[index].get();
    }

    const gert::CompileTimeTensorDesc *GetOutputDesc(uint32_t index) const
    {
        return outputDesc_[index].get();
    }

    const gert::StorageShape *GetOutputShape(uint32_t index) const
    {
        return &outputShape_[index];
    }

    const gert::Tensor *GetOutputTensor(uint32_t index) const
    {
        return outputTensor_[index].get();
    }

    const char *GetNodeName() const
    {
        return nodeName_.c_str();
    }

    const std::shared_ptr<RuntimeAttrs> &GetAttrs() const
    {
        return runtimeAttrs_;
    }

    void AddInputDesc(std::shared_ptr<gert::CompileTimeTensorDesc> desc)
    {
        inputDesc_.push_back(desc);
    }

    void AddOutputDesc(std::shared_ptr<gert::CompileTimeTensorDesc> desc)
    {
        outputDesc_.push_back(desc);
    }

    void SetAttrs(std::shared_ptr<RuntimeAttrs> runtimeAttrs)
    {
        runtimeAttrs_ = runtimeAttrs;
    }

    void SetWorkspaceSizes(size_t userSize)
    {
        auto platformAscendC = platform_ascendc::PlatformAscendCManager::GetInstance();
        systemWorkSpaceSize_ = static_cast<size_t>(platformAscendC->GetLibApiWorkSpaceSize());
        userWorkSpaceSize_ = userSize;
    }

    size_t *GetWorkspaceSizes(uint32_t index)
    {
        return workSpaceSize_[index];
    }

    // Must be called after SetWorkspaceSizes()
    size_t GetWorkspaceSize()
    {
        return systemWorkSpaceSize_ + userWorkSpaceSize_;
    }

    template <typename T>
    at::Tensor GetTilingTensor(const T &tilingData) const
    {
        return TilingTensorCache<T>::Get(tilingData, nodeName_);
    }

    // Deleted, do not need to use these functions
    void SetBlockDim(int blockDim) = delete;
    void SetTilingKey(int tilingKey) = delete;
    gert::TilingData *GetRawTilingData() const = delete;

private:
    // init from user definition
    // input include input and optional input (for adapt aclnn)
    std::vector<std::shared_ptr<gert::CompileTimeTensorDesc>> inputDesc_;
    std::vector<std::shared_ptr<gert::CompileTimeTensorDesc>> outputDesc_;
    std::shared_ptr<RuntimeAttrs> runtimeAttrs_;

    // init from registry
    std::vector<std::shared_ptr<gert::Tensor>> inputTensor_;
    std::vector<std::shared_ptr<gert::Tensor>> outputTensor_;
    std::vector<gert::StorageShape> inputShape_;
    std::vector<gert::StorageShape> outputShape_;

    std::string nodeName_;
    gert::TilingData *rawTilingData_ = nullptr;
    size_t systemWorkSpaceSize_ = 0;
    size_t userWorkSpaceSize_ = 0;
    std::vector<size_t *> workSpaceSize_{&systemWorkSpaceSize_, &userWorkSpaceSize_};
};

// TODO: Do automatic registry template class at compile time
class OpDef
{
public:
    using OutputDef = InputDef;
    explicit OpDef(const std::string &name) : opName_(name) {}

    InputDef &Input(const std::string &name)
    {
        inputs_.emplace_back(name, InputDef());
        return inputs_.back().second;
    }

    AttrDef &Attr(const std::string &name)
    {
        attrs_.emplace_back(name, AttrDef());
        return attrs_.back().second;
    }

    InputDef &Output(const std::string &name)
    {
        outputs_.emplace_back(name, OutputDef());
        return outputs_.back().second;
    }

    void SetAttrStr(const std::string attrName, std::string strVal)
    {
        for (auto &pair : attrs_) {
            if (pair.first == attrName) {
                pair.second.SetStr(strVal);
                return;
            }
        }
        throw std::runtime_error("[GE_Helper] SetAttrStr failed, attrName not exists");
    }

    void SetAttrAny(const std::string attrName, std::any anyVal)
    {
        for (auto &pair : attrs_) {
            if (pair.first == attrName) {
                pair.second.SetAny(anyVal);
                return;
            }
        }
        throw std::runtime_error("[GE_Helper] SetAttrAny failed, attrName not exists");
    }

    void SetToContext(std::shared_ptr<TilingContext> &context, at::ScalarType &scalarType)
    {
        auto geType = SCALAR_TYPE_TO_GE_DATATYPE(scalarType);
        TORCH_CHECK(!inputs_.empty(), "[GE_Helper] SetToContext: Check the op definition file");

        const auto &firstParamTypes = inputs_[0].second.GetDataTypes();
        auto it = std::find(firstParamTypes.begin(), firstParamTypes.end(), geType);
        TORCH_CHECK(it != firstParamTypes.end(),
                    "[GE_Helper] SetToContext: Invalid input type, please check the op definition file");
        uint32_t index = std::distance(firstParamTypes.begin(), it);

        for (auto &input : inputs_) {
            auto tensorDesc = std::make_shared<gert::CompileTimeTensorDesc>();
            tensorDesc->SetDataType(input.second.GetDataType(index));
            tensorDesc->SetOriginFormat(input.second.GetFormat(index));
            context->AddInputDesc(tensorDesc);
        }

        for (auto &output : outputs_) {
            auto tensorDesc = std::make_shared<gert::CompileTimeTensorDesc>();
            tensorDesc->SetDataType(output.second.GetDataType(index));
            tensorDesc->SetOriginFormat(output.second.GetFormat(index));
            context->AddOutputDesc(tensorDesc);
        }

        auto runtimeAttrs = std::make_shared<RuntimeAttrs>();
        for (auto &attr : attrs_) {
            if (attr.second.IsString()) {
                runtimeAttrs->AddStr(attr.second.GetString());
                runtimeAttrs->AddAny(std::any{});
            } else {
                runtimeAttrs->AddAny(attr.second.GetValue());
                runtimeAttrs->AddStr("");
            }
        }
        context->SetAttrs(runtimeAttrs);
    }

    const AttrDef GetAttr(uint32_t index) const
    {
        return attrs_[index].second;
    }

private:
    std::string opName_;
    std::vector<std::pair<std::string, InputDef>> inputs_;
    std::vector<std::pair<std::string, OutputDef>> outputs_;
    std::vector<std::pair<std::string, AttrDef>> attrs_;
};

inline gert::StorageShape CreateStorageShape(const std::vector<int64_t> origin, const std::vector<int64_t> storage)
{
    TORCH_CHECK(origin.size() <= 4 && origin.size() == storage.size(),
                "[GE_Helper] CreateStorageShape: Unsupported vector size");
    switch (origin.size()) {
        case DIM0:
            return gert::StorageShape({}, {});
        case DIM1:
            return gert::StorageShape({origin[0]}, {storage[0]});
        case DIM2:
            return gert::StorageShape({origin[0], origin[1]}, {storage[0], storage[1]});
        case DIM3:
            return gert::StorageShape({origin[0], origin[1], origin[2]}, {storage[0], storage[1], storage[2]});
        case DIM4:
            return gert::StorageShape({origin[0], origin[1], origin[2], origin[3]},
                                      {storage[0], storage[1], storage[2], storage[3]});
    }
    return gert::StorageShape({}, {});
}
}  // namespace ge_helper
}  // namespace sglang
#endif
