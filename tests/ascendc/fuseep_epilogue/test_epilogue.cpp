// Diagnostic for the actual FuseEP GEMM2 epilogue. Only the peer-memory
// transport is replaced with local GM; the production epilogue is included.
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "const_args.hpp"
#include "dispatch_policy_custom.hpp"
#include "kernel_operator.h"

#define SYNC_UTIL_HPP
class HcclShmem {
public:
  GM_ADDR window;
  __aicore__ inline GM_ADDR operator()(int64_t offset, int32_t) const {
    return window + offset;
  }
};
using namespace AscendC;
#include "block_epilogue_pertoken_v2.hpp"

extern "C" __global__ __aicore__ void
epilogue_probe(GM_ADDR input, GM_ADDR scales, GM_ADDR counts, GM_ADDR prefix,
               GM_ADDR output, uint32_t rows, uint32_t columns,
               uint32_t narrowMask) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace Catlass;
  using ArchTag = DispatchFfnArch;
  using C = Gemm::GemmType<half, layout::RowMajor>;
  using D = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
  using S = Gemm::GemmType<uint64_t, layout::VectorLayout>;
  using TS = Gemm::GemmType<float, layout::VectorLayout>;
  using Copy = Epilogue::Tile::TileCopy<ArchTag, C, S, TS, D>;
  using Epilogue = Epilogue::Block::BlockEpilogue<
      Epilogue::EpilogueDispatchFfnPerTokenDequant<2>, C, TS, D, Copy>;
  Catlass::Arch::Resource<ArchTag> resource;
  HcclShmem local{output};
  typename Epilogue::Params params{1,
                                   1,
                                   0,
                                   (__gm__ int32_t *)counts,
                                   layout::RowMajor{rows, columns},
                                   (int32_t)columns,
                                   256,
                                   local,
                                   0};
  Epilogue epilogue(resource, params);
  auto poison = resource.ubBuf.template GetBufferByByte<uint32_t>(0);
  Duplicate(poison, uint32_t(0x7fc17fc1), 36 * 1024);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
  SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
  SetFlag<HardEvent::S_MTE2>(EVENT_ID3);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
  SetMaskNorm();
  SetVectorMask<float>(0, narrowMask ? uint64_t(0xff) : uint64_t(-1));
  GlobalTensor<half> gmInput;
  gmInput.SetGlobalBuffer((__gm__ half *)input);
  GlobalTensor<float> gmScale;
  gmScale.SetGlobalBuffer((__gm__ float *)scales);
  GlobalTensor<int32_t> gmPrefix;
  gmPrefix.SetGlobalBuffer((__gm__ int32_t *)prefix);
  GemmCoord coord{0, 0, 0};
  GemmCoord shape{rows, columns, 1};
  epilogue(gmInput, gmScale, coord, shape, 0, 0, gmPrefix);
  epilogue.Finalize();
  ResetMask();
  PipeBarrier<PIPE_ALL>();
}

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
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

int main(int argc, char **argv) {
  const int device = argc > 1 ? std::atoi(argv[1]) : 0;
  ACL_OK(aclInit(nullptr));
  ACL_OK(aclrtSetDevice(device));
  aclrtStream stream;
  ACL_OK(aclrtCreateStream(&stream));
  int failedCases = 0;
  for (uint32_t rows : {1, 3, 16}) {
    for (uint32_t columns : {64, 128, 256}) {
      for (uint32_t narrowMask : {0, 1}) {
        uint16_t halfBits[8] = {0x3c00, 0x4000, 0x4200, 0x4400,
                                0x4500, 0x4600, 0x4700, 0x4800};
        std::vector<uint16_t> input(rows * columns), output(rows * columns);
        std::vector<float> scales(rows);
        for (uint32_t row = 0; row < rows; ++row) {
          scales[row] = row + 1;
          for (uint32_t col = 0; col < columns; ++col)
            input[row * columns + col] = halfBits[col % 8];
        }
        int32_t counts = rows, prefix = 0;
        void *dx, *ds, *dc, *dp, *dy;
        size_t bytes = input.size() * sizeof(uint16_t);
        ACL_OK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_OK(
            aclrtMalloc(&ds, rows * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_OK(aclrtMalloc(&dc, sizeof(counts), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_OK(aclrtMalloc(&dp, sizeof(prefix), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_OK(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_OK(aclrtMemcpy(dx, bytes, input.data(), bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_OK(aclrtMemcpy(ds, rows * sizeof(float), scales.data(),
                           rows * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_OK(aclrtMemcpy(dc, sizeof(counts), &counts, sizeof(counts),
                           ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_OK(aclrtMemcpy(dp, sizeof(prefix), &prefix, sizeof(prefix),
                           ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_OK(aclrtMemset(dy, bytes, 0xff, bytes));
        epilogue_probe<<<1, nullptr, stream>>>(
            (GM_ADDR)dx, (GM_ADDR)ds, (GM_ADDR)dc, (GM_ADDR)dp, (GM_ADDR)dy,
            rows, columns, narrowMask);
        ACL_OK(aclrtSynchronizeStream(stream));
        ACL_OK(aclrtMemcpy(output.data(), bytes, dy, bytes,
                           ACL_MEMCPY_DEVICE_TO_HOST));
        size_t nonfinite = 0, mismatch = 0;
        for (size_t i = 0; i < output.size(); ++i) {
          uint32_t bits = uint32_t(output[i]) << 16;
          float actual;
          std::memcpy(&actual, &bits, sizeof(actual));
          float expected = float(i % columns % 8 + 1) * float(i / columns + 1);
          nonfinite += !std::isfinite(actual);
          mismatch += actual != expected;
        }
        std::printf(
            "rows=%u cols=%u narrow_mask=%u nonfinite=%zu mismatch=%zu\n", rows,
            columns, narrowMask, nonfinite, mismatch);
        failedCases += mismatch != 0;
        ACL_OK(aclrtFree(dx));
        ACL_OK(aclrtFree(ds));
        ACL_OK(aclrtFree(dc));
        ACL_OK(aclrtFree(dp));
        ACL_OK(aclrtFree(dy));
      }
    }
  }
  ACL_OK(aclrtDestroyStream(stream));
  ACL_OK(aclrtResetDevice(device));
  ACL_OK(aclFinalize());
  return failedCases ? 1 : 0;
}
#endif
