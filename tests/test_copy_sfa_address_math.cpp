#include <cassert>
#include <cstdint>

#include "../csrc/fused_scatter_copy_sparse_flash_attention/op_kernel/sparse_flash_attention_impl/copy_sfa_address_math.h"

using sglang::npu_kernel::copy_sfa::SourceGapBytes;

namespace {

int64_t PhysicalAddress(int64_t token, int64_t rowStride, int64_t planeOffset,
                        int64_t elementBytes) {
  return (token * rowStride + planeOffset) * elementBytes;
}

void CheckPair(int64_t token0, int64_t token1, int64_t logicalWidth) {
  constexpr int64_t kRowStride = 576;
  constexpr int64_t kElementBytes = 2;
  const int64_t gap = SourceGapBytes(token1 - token0, 1, kRowStride,
                                     logicalWidth, kElementBytes);
  const int64_t firstStart =
      PhysicalAddress(token0, kRowStride, 0, kElementBytes);
  const int64_t secondStart =
      PhysicalAddress(token1, kRowStride, 0, kElementBytes);
  assert(gap == secondStart - firstStart - logicalWidth * kElementBytes);
  assert(gap >= 0);
}

} // namespace

int main() {
  // Adjacent rows: the combined row's unused plane is part of the gap.
  assert(SourceGapBytes(1, 1, 576, 512, 2) == 128); // CKV
  assert(SourceGapBytes(1, 1, 576, 64, 2) == 1024); // KPE

  // Nonadjacent rows retain the complete physical row spacing.
  CheckPair(10, 13, 512);
  CheckPair(10, 13, 64);

  // Cross-page: the physical page-table mapping places logical neighbors
  // in pages 3 and 5, so their physical token coordinates differ by 129.
  constexpr int64_t kToken0 = 3 * 128 + 127;
  constexpr int64_t kToken1 = 5 * 128;
  CheckPair(kToken0, kToken1, 512);
  CheckPair(kToken0, kToken1, 64);

  // Split storage remains algebraically identical to the original formula.
  assert(SourceGapBytes(5, 2, 512, 512, 2) == (5 - 2) * 512 * 2);
  assert(SourceGapBytes(5, 2, 64, 64, 2) == (5 - 2) * 64 * 2);
}
