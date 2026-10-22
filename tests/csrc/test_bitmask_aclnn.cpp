#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SGL_BITMASK_HOST_TEST_ONLY
#include <dlfcn.h>
#include <filesystem>
#include "acl/acl.h"
#include SGL_BITMASK_ACLNN_HEADER
#endif

namespace {

enum class Dtype { FP32, FP16, BF16 };
enum class Mask { None, All, Alternating, Random };
constexpr std::array<std::array<int64_t, 2>, 4> shapes{{{1, 64}, {4, 320}, {64, 32000}, {2, 128256}}};
constexpr std::array<Dtype, 3> dtypes{Dtype::FP32, Dtype::FP16, Dtype::BF16};
constexpr std::array<Mask, 4> masks{Mask::None, Mask::All, Mask::Alternating, Mask::Random};
constexpr uint32_t seed = 20261022;

size_t ElementSize(Dtype type) { return type == Dtype::FP32 ? 4 : 2; }
const char* Name(Dtype type) { return type == Dtype::FP32 ? "fp32" : type == Dtype::FP16 ? "fp16" : "bf16"; }
const char* Name(Mask mask)
{
    return mask == Mask::None ? "all_unmasked" : mask == Mask::All ? "all_masked" :
           mask == Mask::Alternating ? "half_masked" : "random";
}
#ifdef SGL_BITMASK_HOST_TEST_ONLY
uint32_t NegativeInfinity(Dtype type)
{
    return type == Dtype::FP32 ? 0xff800000U : type == Dtype::FP16 ? 0xfc00U : 0xff80U;
}
#endif

void PutBits(std::vector<uint8_t>& data, size_t index, Dtype type, uint32_t bits)
{
    if (type == Dtype::FP32) {
        std::memcpy(data.data() + index * 4, &bits, 4);
    } else {
        const auto shortBits = static_cast<uint16_t>(bits);
        std::memcpy(data.data() + index * 2, &shortBits, 2);
    }
}

float Value(const std::vector<uint8_t>& data, size_t index, Dtype type)
{
    uint32_t bits = 0;
    if (type == Dtype::FP32) {
        std::memcpy(&bits, data.data() + index * 4, 4);
    } else {
        uint16_t shortBits = 0;
        std::memcpy(&shortBits, data.data() + index * 2, 2);
        if (type == Dtype::BF16) {
            bits = static_cast<uint32_t>(shortBits) << 16;
        } else {
            const int exponent = (shortBits >> 10) & 31;
            const int fraction = shortBits & 1023;
            const float magnitude = exponent == 31 ? (fraction ? std::numeric_limits<float>::quiet_NaN() :
                std::numeric_limits<float>::infinity()) :
                std::ldexp(static_cast<float>(exponent ? 1024 + fraction : fraction),
                           exponent ? exponent - 25 : -24);
            return (shortBits & 0x8000) ? -magnitude : magnitude;
        }
    }
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

struct Case {
    int64_t rows;
    int64_t width;
    int64_t padded;
    Dtype type;
    Mask mask;
    std::vector<uint8_t> logits;
    std::vector<uint32_t> bitmask;

    std::string Label() const
    {
        return std::to_string(rows) + "x" + std::to_string(width) + "/" + Name(type) + "/" + Name(mask);
    }
};

Case MakeCase(int64_t rows, int64_t width, Dtype type, Mask mask)
{
    const int64_t padded = (width + 255) / 256 * 256;
    Case c{rows, width, padded, type, mask,
           std::vector<uint8_t>(rows * padded * ElementSize(type), 0),
           std::vector<uint32_t>(rows * (padded / 32), 0)};
    std::mt19937 values(seed), maskValues(seed + 1);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            // Finite IEEE values, with random sign, exponent and fraction, directly in each dtype.
            const uint32_t random = values();
            const uint32_t exponent = 121 + values() % 12;
            uint32_t bits = (random & 0x807fffffU) | (exponent << 23);
            if (type == Dtype::FP16) bits = ((bits >> 16) & 0x8000U) | ((exponent - 112) << 10) | (random & 1023);
            if (type == Dtype::BF16) bits >>= 16;
            PutBits(c.logits, row * padded + col, type, bits);
        }
        for (int64_t word = 0; word < (width + 31) / 32; ++word) {
            uint32_t bits = mask == Mask::None ? 0xffffffffU : mask == Mask::All ? 0 :
                mask == Mask::Alternating ? (word % 2 ? 0xaaaaaaaaU : 0x55555555U) : maskValues();
            if (word == width / 32 && width % 32) bits &= (uint32_t{1} << (width % 32)) - 1;
            c.bitmask[row * (padded / 32) + word] = bits;
        }
    }
    return c;
}

bool Verify(const Case& c, const std::vector<uint8_t>& output, const std::vector<uint32_t>& bitmask,
            std::string& error)
{
    if (output.size() != c.logits.size() || bitmask != c.bitmask) {
        error = "wrong output size or bitmask input modified";
        return false;
    }
    // Also check the padded tail, whose logits and mask are zero before execution.
    for (int64_t row = 0; row < c.rows; ++row) {
        for (int64_t col = 0; col < c.padded; ++col) {
            const size_t i = row * c.padded + col;
            const bool keep = (c.bitmask[row * (c.padded / 32) + col / 32] >> (col % 32)) & 1U;
            const float actual = Value(output, i, c.type);
            const float expected = Value(c.logits, i, c.type);
            const bool ok = keep ? (std::isfinite(actual) && std::fabs(actual - expected) <= 1e-2f + 1e-2f * std::fabs(expected)) :
                                   (std::isinf(actual) && actual < 0);
            if (!ok) {
                error = "row=" + std::to_string(row) + " col=" + std::to_string(col) +
                    " actual=" + std::to_string(actual) + " expected=" + (keep ? std::to_string(expected) : "-inf");
                return false;
            }
        }
    }
    return true;
}

#ifdef SGL_BITMASK_HOST_TEST_ONLY
void CheckReference()
{
    for (Dtype type : dtypes) {
        std::vector<uint8_t> known(4 * ElementSize(type));
        PutBits(known, 0, type, type == Dtype::FP32 ? 0x3f800000U : type == Dtype::FP16 ? 0x3c00U : 0x3f80U);
        PutBits(known, 1, type, type == Dtype::FP32 ? 0xc0000000U : 0xc000U);
        PutBits(known, 2, type, NegativeInfinity(type));
        PutBits(known, 3, type, NegativeInfinity(type) | 1U);
        if (Value(known, 0, type) != 1.0f || Value(known, 1, type) != -2.0f ||
            Value(known, 2, type) != -std::numeric_limits<float>::infinity() || !std::isnan(Value(known, 3, type)))
            throw std::runtime_error("dtype decoding fixture failed");
        auto c = MakeCase(1, 64, type, Mask::Alternating);
        std::string error;
        auto output = c.logits;
        // Hand-specified alternating pattern: word 0 keeps even columns, word 1 keeps odd columns.
        for (size_t i = 0; i < 256; ++i) {
            const bool keep = i < 64 && ((i < 32 && i % 2 == 0) || (i >= 32 && i % 2 == 1));
            if (!keep) PutBits(output, i, type, NegativeInfinity(type));
        }
        if (!Verify(c, output, c.bitmask, error)) throw std::runtime_error("reference fixture: " + error);
        auto wrong = output;
        PutBits(wrong, 1, type, 0);
        if (Verify(c, wrong, c.bitmask, error)) throw std::runtime_error("missed unmasked output");
        wrong = output;
        PutBits(wrong, 0, type, NegativeInfinity(type));
        if (Verify(c, wrong, c.bitmask, error)) throw std::runtime_error("missed damaged retained value");
        PutBits(wrong, 0, type, NegativeInfinity(type) | 1U);
        if (Verify(c, wrong, c.bitmask, error)) throw std::runtime_error("missed NaN in retained value");
        wrong = output;
        PutBits(wrong, 64, type, 0);
        if (Verify(c, wrong, c.bitmask, error)) throw std::runtime_error("missed damaged padding");
        auto changedMask = c.bitmask;
        changedMask[0] ^= 1;
        if (Verify(c, output, changedMask, error)) throw std::runtime_error("missed bitmask mutation");
    }
    size_t count = 0;
    for (auto shape : shapes) for (Dtype type : dtypes) for (Mask mask : masks) {
        const auto c = MakeCase(shape[0], shape[1], type, mask);
        if (c.logits != MakeCase(shape[0], shape[1], type, mask).logits || c.padded % 256)
            throw std::runtime_error("non-reproducible or unaligned test input");
        for (size_t i = 0; i < c.logits.size() / ElementSize(type); ++i)
            if (!std::isfinite(Value(c.logits, i, type))) throw std::runtime_error("nonfinite generated input");
        ++count;
    }
    std::cout << "PASS: Host test-data checks for " << count << " cases and comparison fault checks; no CANN/NPU execution\n";
}
#else
void Check(int64_t status, const char* operation)
{
    if (status == 0) return;
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status) +
                             (detail ? std::string("; ") + detail : ""));
}

struct Session {
    bool initialized = false;
    bool selected = false;
    aclrtStream stream = nullptr;
    void Open()
    {
        Check(aclInit(nullptr), "aclInit");
        initialized = true;
        Check(aclrtSetDevice(0), "aclrtSetDevice(0)");
        selected = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Session()
    {
        if (stream) { aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); }
        if (selected) aclrtResetDevice(0);
        if (initialized) aclFinalize();
    }
};

struct DeviceData {
    void* logits = nullptr;
    void* mask = nullptr;
    void* workspace = nullptr;
    aclTensor* logitsTensor = nullptr;
    aclTensor* maskTensor = nullptr;
    aclrtStream stream;
    explicit DeviceData(aclrtStream value) : stream(value) {}
    ~DeviceData()
    {
        // Keep all buffers alive until any submitted work has finished, including error paths.
        aclrtSynchronizeStream(stream);
        if (logitsTensor) aclDestroyTensor(logitsTensor);
        if (maskTensor) aclDestroyTensor(maskTensor);
        if (workspace) aclrtFree(workspace);
        if (mask) aclrtFree(mask);
        if (logits) aclrtFree(logits);
    }
};

void RunCase(const Case& c, aclrtStream stream)
{
    DeviceData data(stream);
    const size_t maskBytes = c.bitmask.size() * sizeof(uint32_t);
    Check(aclrtMalloc(&data.logits, c.logits.size(), ACL_MEM_MALLOC_HUGE_FIRST), "allocate logits");
    Check(aclrtMalloc(&data.mask, maskBytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate bitmask");
    Check(aclrtMemcpy(data.logits, c.logits.size(), c.logits.data(), c.logits.size(), ACL_MEMCPY_HOST_TO_DEVICE), "upload logits");
    Check(aclrtMemcpy(data.mask, maskBytes, c.bitmask.data(), maskBytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload bitmask");
    const int64_t logitsShape[]{c.rows, c.padded}, logitsStrides[]{c.padded, 1};
    const int64_t maskShape[]{c.rows, c.padded / 32}, maskStrides[]{c.padded / 32, 1};
    const aclDataType dtype = c.type == Dtype::FP32 ? ACL_FLOAT : c.type == Dtype::FP16 ? ACL_FLOAT16 : ACL_BF16;
    data.logitsTensor = aclCreateTensor(logitsShape, 2, dtype, logitsStrides, 0, ACL_FORMAT_ND, logitsShape, 2, data.logits);
    data.maskTensor = aclCreateTensor(maskShape, 2, ACL_INT32, maskStrides, 0, ACL_FORMAT_ND, maskShape, 2, data.mask);
    if (!data.logitsTensor || !data.maskTensor) throw std::runtime_error("aclCreateTensor returned null");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    Check(aclnnSglApplyTokenBitMaskGetWorkspaceSize(data.logitsTensor, data.maskTensor, &workspaceSize, &executor), "GetWorkspaceSize");
    if (workspaceSize) Check(aclrtMalloc(&data.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "allocate workspace");
    Check(aclnnSglApplyTokenBitMask(data.workspace, workspaceSize, executor, stream), "aclnnSglApplyTokenBitMask");
    Check(aclrtSynchronizeStream(stream), "wait for Kernel");
    std::vector<uint8_t> result(c.logits.size());
    std::vector<uint32_t> returnedMask(c.bitmask.size());
    // Read the original allocation: a separate output tensor cannot satisfy this check.
    Check(aclrtMemcpy(result.data(), result.size(), data.logits, result.size(), ACL_MEMCPY_DEVICE_TO_HOST), "read original logits allocation");
    Check(aclrtMemcpy(returnedMask.data(), maskBytes, data.mask, maskBytes, ACL_MEMCPY_DEVICE_TO_HOST), "read bitmask");
    std::string error;
    if (!Verify(c, result, returnedMask, error)) throw std::runtime_error(c.Label() + ": " + error);
    std::cout << "PASS: " << c.Label() << " padded=" << c.padded
              << " inplace=yes bitmask_unchanged=yes workspace_bytes=" << workspaceSize << std::endl;
}

void RunNpu()
{
    const char* visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (!visible || !*visible || std::string(visible).find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Set ASCEND_RT_VISIBLE_DEVICES to one available device ID before running");
    const char* vendor = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (!vendor || !std::filesystem::equivalent(std::filesystem::path(vendor) / "op_api/lib/libcust_opapi.so", SGL_BITMASK_OPAPI_PATH))
        throw std::runtime_error("ASCEND_CUSTOM_OPP_PATH must select the installed vendor used at compile time");
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&aclnnSglApplyTokenBitMaskGetWorkspaceSize), &info) || !info.dli_fname ||
        !std::filesystem::equivalent(info.dli_fname, SGL_BITMASK_OPAPI_PATH))
        throw std::runtime_error("ACLNN resolved from a different operator library");
    std::cout << "Loaded operator library: " << info.dli_fname << "\nVendor: " << vendor
              << "\nVisible device: " << visible << "; ACL device index: 0; seed=" << seed << std::endl;
    Session session;
    session.Open();
    size_t passed = 0;
    for (auto shape : shapes) for (Dtype type : dtypes) for (Mask mask : masks) {
        const auto c = MakeCase(shape[0], shape[1], type, mask);
        std::cout << "RUN: " << c.Label() << std::endl;
        RunCase(c, session.stream);
        ++passed;
    }
    std::cout << "PASS: " << passed << "/48 ACLNN cases; numerical results, in-place logits and unchanged bitmask\n"
              << "Torch, indices, old/new tiling equality and performance were not tested." << std::endl;
}
#endif

}  // namespace

int main()
{
    try {
#ifdef SGL_BITMASK_HOST_TEST_ONLY
        CheckReference();
#else
        RunNpu();
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
