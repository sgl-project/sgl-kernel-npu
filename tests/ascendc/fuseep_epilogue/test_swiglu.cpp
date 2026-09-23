// Exercise the production SwiGLU/requantization stage without HCCL or a wheel.
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "dispatch_policy_custom.hpp"
#include "kernel_operator.h"
using namespace AscendC;
#include "block_epilogue_pertoken_swiglu.hpp"

extern "C" __global__ __aicore__ void
swiglu_probe(GM_ADDR input, GM_ADDR scales, GM_ADDR output,
             GM_ADDR outputScales, uint32_t rows, uint32_t columns,
             uint32_t narrowMask) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if ASCEND_IS_AIV {
    using namespace Catlass;
    using ArchTag = DispatchFfnArch;
    using C = Gemm::GemmType<half, layout::RowMajor>;
    using D = Gemm::GemmType<int8_t, layout::RowMajor>;
    using S = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using TS = Gemm::GemmType<float, layout::VectorLayout>;
    using Copy = Epilogue::Tile::TileCopy<ArchTag, C, S, TS, D>;
    using Block = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueDispatchFfnPerTokenDequantSwigluQuant<2>, C, TS, D,
        void, Copy>;
    Arch::Resource<ArchTag> resource;
    Block epilogue(resource, columns);
    auto poison = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
    Duplicate(poison, uint32_t(0x7fc17fc1), 180 * 1024 / sizeof(uint32_t));
    PipeBarrier<PIPE_ALL>();
    SetMaskNorm();
    SetVectorMask<float>(0, narrowMask ? uint64_t(0xff) : uint64_t(-1));
    GlobalTensor<half> gmInput;
    GlobalTensor<float> gmScale, gmOutputScale;
    GlobalTensor<int8_t> gmOutput;
    gmInput.SetGlobalBuffer((__gm__ half *)input);
    gmScale.SetGlobalBuffer((__gm__ float *)scales);
    gmOutput.SetGlobalBuffer((__gm__ int8_t *)output);
    gmOutputScale.SetGlobalBuffer((__gm__ float *)outputScales);
    epilogue(gmInput, MatrixCoord{rows, columns}, gmScale, gmOutput,
             gmOutputScale, 2);
    epilogue.Finalize();
    ResetMask();
    PipeBarrier<PIPE_ALL>();
  }
}

#include "acl/acl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ACL_OK(call)                                                           \
  do {                                                                         \
    auto status = (call);                                                      \
    if (status != ACL_SUCCESS) {                                               \
      std::fprintf(stderr, "%s failed: %d\n", #call, int(status));             \
      return 2;                                                                \
    }                                                                          \
  } while (0)

// The generated values are exact FP16 normal numbers or zero.
uint16_t halfBits(float value) {
  if (value == 0)
    return 0;
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return ((bits >> 16) & 0x8000) | ((((bits >> 23) & 0xff) - 112) << 10) |
         ((bits >> 13) & 0x3ff);
}

int main(int argc, char **argv) {
  const int device = argc > 1 ? std::atoi(argv[1]) : 0;
  ACL_OK(aclInit(nullptr));
  ACL_OK(aclrtSetDevice(device));
  aclrtStream stream;
  ACL_OK(aclrtCreateStream(&stream));
  int failures = 0;
  for (uint32_t rows : {1, 3, 17}) {
    for (uint32_t columns : {256, 1536, 7168}) {
      const uint32_t width = columns / 2;
      std::vector<uint16_t> input(rows * columns);
      std::vector<float> scales(rows), outputScales(rows),
          expected(rows * width);
      std::vector<float> expectedScales(rows);
      std::vector<int8_t> output(rows * width);
      for (uint32_t row = 0; row < rows; ++row) {
        scales[row] = std::ldexp(1.0f, int(row % 4) - 3);
        float maxValue = 0;
        for (uint32_t col = 0; col < width; ++col) {
          float gate = float(int((col * 7 + row) % 65) - 32) / 8;
          float up = float(int((col * 11 + row) % 63) - 31) / 8;
          input[row * columns + col] = halfBits(gate);
          input[row * columns + width + col] = halfBits(up);
          gate *= scales[row];
          up *= scales[row];
          float value = gate / (1 + std::exp(-gate)) * up;
          expected[row * width + col] = value;
          maxValue = std::max(maxValue, std::abs(value));
        }
        expectedScales[row] = maxValue / 127;
      }
      void *dx, *ds, *dy, *dys;
      const size_t xb = input.size() * 2, sb = rows * sizeof(float);
      const size_t yb = output.size();
      ACL_OK(aclrtMalloc(&dx, xb, ACL_MEM_MALLOC_HUGE_FIRST));
      ACL_OK(aclrtMalloc(&ds, sb, ACL_MEM_MALLOC_HUGE_FIRST));
      ACL_OK(aclrtMalloc(&dy, yb, ACL_MEM_MALLOC_HUGE_FIRST));
      ACL_OK(aclrtMalloc(&dys, sb, ACL_MEM_MALLOC_HUGE_FIRST));
      ACL_OK(aclrtMemcpy(dx, xb, input.data(), xb, ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_OK(aclrtMemcpy(ds, sb, scales.data(), sb, ACL_MEMCPY_HOST_TO_DEVICE));
      for (uint32_t narrowMask : {0, 1}) {
        for (int iteration = 0; iteration < 3; ++iteration) {
          ACL_OK(aclrtMemset(dy, yb, 0x80, yb));
          ACL_OK(aclrtMemset(dys, sb, 0xff, sb));
          swiglu_probe<<<1, nullptr, stream>>>((GM_ADDR)dx, (GM_ADDR)ds,
                                               (GM_ADDR)dy, (GM_ADDR)dys, rows,
                                               columns, narrowMask);
          ACL_OK(aclrtSynchronizeStream(stream));
          ACL_OK(aclrtMemcpy(output.data(), yb, dy, yb,
                             ACL_MEMCPY_DEVICE_TO_HOST));
          ACL_OK(aclrtMemcpy(outputScales.data(), sb, dys, sb,
                             ACL_MEMCPY_DEVICE_TO_HOST));
          size_t badScale = 0, badQuant = 0;
          int maxQuantError = 0;
          for (uint32_t row = 0; row < rows; ++row) {
            const float scale = outputScales[row];
            badScale +=
                !std::isfinite(scale) || std::abs(scale - expectedScales[row]) >
                                             expectedScales[row] * 2e-5f;
            for (uint32_t col = 0; col < width; ++col) {
              const size_t i = row * width + col;
              const int ref =
                  int(std::nearbyint(expected[i] / expectedScales[row]));
              const int error = std::abs(int(output[i]) - ref);
              // Independent CPU math can straddle a quantization midpoint.
              badQuant += error > 1;
              maxQuantError = std::max(maxQuantError, error);
            }
          }
          std::printf("rows=%u columns=%u narrow_mask=%u iteration=%d "
                      "bad_scale=%zu bad_quant=%zu max_quant_error=%d\n",
                      rows, columns, narrowMask, iteration, badScale, badQuant,
                      maxQuantError);
          failures += badScale + badQuant > 0;
        }
      }
      ACL_OK(aclrtFree(dx));
      ACL_OK(aclrtFree(ds));
      ACL_OK(aclrtFree(dy));
      ACL_OK(aclrtFree(dys));
    }
  }
  ACL_OK(aclrtDestroyStream(stream));
  ACL_OK(aclrtResetDevice(device));
  ACL_OK(aclFinalize());
  return failures ? 1 : 0;
}
