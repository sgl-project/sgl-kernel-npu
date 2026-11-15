#include <cassert>
#include <cstdint>

#include "../csrc/attention/fused_quant_scatter_copy_sparse_flash_attention/op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"

using fused_quant_scatter_copy_sfa::RingCacheTokens;
using fused_quant_scatter_copy_sfa::RingPrefix;
using fused_quant_scatter_copy_sfa::RingTailLength;

int main() {
  constexpr int64_t capacity = 8192;
  assert(RingPrefix(128) == 128);
  assert(RingPrefix(255) == 128);
  assert(RingPrefix(256) == 256);

  // Q1 at a page boundary exposes one current row after the resident C rows.
  assert(RingTailLength(256, 257) == 1);
  assert(RingCacheTokens(capacity, 256, 257) == capacity + 1);

  // Q4 may remain in one page or cross into the next page. The cache length
  // is always C + (last_position + 1 - floor(first / 128) * 128).
  assert(RingTailLength(260, 264) == 8);
  assert(RingCacheTokens(capacity, 260, 264) == capacity + 8);
  assert(RingTailLength(126, 130) == 130);
  assert(RingCacheTokens(capacity, 126, 130) == capacity + 130);
  return 0;
}
