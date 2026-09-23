// Exercise production routing and dynamic INT8 quantization without HCCL.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The host tiling headers require the standard math/algorithm declarations.
#include "kernel_operator.h"
#include "moe_init_routing_quant_v2_tiling.h"
#include "moe_v2_fullload_dynamic_quant.h"

extern "C" __global__ __aicore__ void
routing_quant_probe(GM_ADDR input, GM_ADDR experts, GM_ADDR quant,
                    GM_ADDR indices, GM_ADDR counts, GM_ADDR workspace,
                    GM_ADDR tilingGM) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if ASCEND_IS_AIV {
    using namespace AscendC;
    optiling::MoeInitRoutingQuantV2TilingData tiling;
    auto *localTiling = reinterpret_cast<uint32_t *>(&tiling);
    auto *globalTiling = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(tiling) / sizeof(uint32_t); ++i) {
      localTiling[i] = globalTiling[i];
    }
    TPipe pipe;
    MoeInitRoutingQuantV2::MoeV2FullLoadDynamicQuant<bfloat16_t> op;
    op.Init(input, experts, quant, indices, counts, nullptr, nullptr, workspace,
            &tiling, &pipe);
    // Give stale UB scale reads a deterministic finite value.
    LocalTensor<float> poison;
    poison.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    poison.address_.bufferAddr = 0;
    poison.address_.dataLen = 180 * 1024;
    Duplicate(poison, 16384.0f, 180 * 1024 / sizeof(float));
    PipeBarrier<PIPE_ALL>();
    op.Process();
    pipe.Destroy();
  }
}

#include "acl/acl.h"
#define CHECK(call)                                                            \
  do {                                                                         \
    auto rc = (call);                                                          \
    if (rc != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "%s failed: %d\n", #call, int(rc));                 \
      return 2;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  int device = argc > 1 ? std::atoi(argv[1]) : 0;
  CHECK(aclInit(nullptr));
  CHECK(aclrtSetDevice(device));
  aclrtStream stream;
  CHECK(aclrtCreateStream(&stream));
  int failures = 0;
  constexpr uint32_t cols = 2048, topk = 8, numExperts = 128,
                     stride = cols + 512;
  for (uint32_t rows : {1, 17, 128}) {
    optiling::MoeInitRoutingQuantV2TilingBase tiler;
    if (!tiler.DoTiling(rows, cols, topk, 0, numExperts, 0, 0, 2, false, 2, 1,
                        0, 2, 192 * 1024 - 256) ||
        tiler.tilingKey_ != 21000) {
      std::fprintf(stderr, "Unexpected routing tiling\n");
      return 2;
    }
    auto tiling = tiler.quantTilingData;
    std::vector<uint16_t> input(rows * cols);
    std::vector<int32_t> experts(rows * topk), indices(rows * topk),
        counts(numExperts);
    std::vector<uint8_t> quant(rows * topk * stride);
    for (uint32_t row = 0; row < rows; ++row) {
      float scale = std::ldexp(1.0f, int(row % 5) - 2);
      for (uint32_t col = 0; col < cols; ++col) {
        float value = float(int((col + row * 13) % 255) - 127) * scale;
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        input[row * cols + col] = bits >> 16;
      }
      for (uint32_t k = 0; k < topk; ++k)
        experts[row * topk + k] = (row * topk + k) % numExperts;
    }
    void *dx, *de, *dq, *di, *dc, *dw, *dt;
    size_t xb = input.size() * 2, eb = experts.size() * 4, qb = quant.size(),
           cb = counts.size() * 4;
    size_t wb = std::max(size_t(tiler.workspaceSize_), size_t(4096));
    CHECK(aclrtMalloc(&dx, xb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&de, eb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dq, qb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&di, eb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dc, cb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dw, wb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dt, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(dx, xb, input.data(), xb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(de, eb, experts.data(), eb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(dt, sizeof(tiling), &tiling, sizeof(tiling),
                      ACL_MEMCPY_HOST_TO_DEVICE));
    for (int iteration = 0; iteration < 3; ++iteration) {
      CHECK(aclrtMemset(dq, qb, 0xee, qb));
      CHECK(aclrtMemset(di, eb, 0xff, eb));
      routing_quant_probe<<<1, nullptr, stream>>>(
          (GM_ADDR)dx, (GM_ADDR)de, (GM_ADDR)dq, (GM_ADDR)di, (GM_ADDR)dc,
          (GM_ADDR)dw, (GM_ADDR)dt);
      CHECK(aclrtSynchronizeStream(stream));
      CHECK(aclrtMemcpy(quant.data(), qb, dq, qb, ACL_MEMCPY_DEVICE_TO_HOST));
      CHECK(aclrtMemcpy(indices.data(), eb, di, eb, ACL_MEMCPY_DEVICE_TO_HOST));
      CHECK(aclrtMemcpy(counts.data(), cb, dc, cb, ACL_MEMCPY_DEVICE_TO_HOST));
      size_t badScale = 0, badQuant = 0, badIndex = 0, badCount = 0;
      std::vector<bool> seen(rows * topk, false);
      std::vector<int32_t> expectedCounts(numExperts, 0),
          prefix(numExperts + 1, 0);
      for (auto expert : experts)
        ++expectedCounts[expert];
      for (uint32_t e = 0; e < numExperts; ++e)
        prefix[e + 1] = prefix[e] + expectedCounts[e];
      float firstScale = 0;
      for (uint32_t i = 0; i < rows * topk; ++i) {
        int32_t dest = indices[i];
        if (dest < 0 || dest >= int32_t(rows * topk)) {
          ++badIndex;
          continue;
        }
        badIndex += seen[dest] || dest < prefix[experts[i]] ||
                    dest >= prefix[experts[i] + 1];
        seen[dest] = true;
        size_t offset = size_t(dest) * stride;
        float actualScale;
        std::memcpy(&actualScale, quant.data() + offset + cols, 4);
        if (i == 0)
          firstScale = actualScale;
        float expectedScale = std::ldexp(1.0f, int((i / topk) % 5) - 2);
        badScale += actualScale != expectedScale;
        for (uint32_t col = 0; col < cols; ++col) {
          int8_t actual = static_cast<int8_t>(quant[offset + col]);
          badQuant += actual != int((col + (i / topk) * 13) % 255) - 127;
        }
      }
      for (uint32_t e = 0; e < numExperts; ++e) {
        badCount += counts[e] != expectedCounts[e];
      }
      std::printf("rows=%u iteration=%d bad_index=%zu bad_count=%zu "
                  "bad_scale=%zu bad_quant=%zu first_scale=%g\n",
                  rows, iteration, badIndex, badCount, badScale, badQuant,
                  firstScale);
      failures += (badIndex + badCount + badScale + badQuant) > 0;
    }
    CHECK(aclrtFree(dx));
    CHECK(aclrtFree(de));
    CHECK(aclrtFree(dq));
    CHECK(aclrtFree(di));
    CHECK(aclrtFree(dc));
    CHECK(aclrtFree(dw));
    CHECK(aclrtFree(dt));
  }
  CHECK(aclrtDestroyStream(stream));
  CHECK(aclrtResetDevice(device));
  CHECK(aclFinalize());
  return failures ? 1 : 0;
}
