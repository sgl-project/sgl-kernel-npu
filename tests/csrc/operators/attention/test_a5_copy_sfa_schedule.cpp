#include <cassert>

#include "fused_quant_scatter_copy_sparse_flash_attention_schedule.h"

using fused_quant_scatter_copy_sfa::PackedCopyMode;
using fused_quant_scatter_copy_sfa::UpdatePackedCopyMode;

int main() {
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 0, 8192, 0, true, true,
                              true) == PackedCopyMode::NONE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 8, 8192, 8, true, true,
                              true) == PackedCopyMode::NONE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 9, 8192, 8, true, true,
                              true) == PackedCopyMode::REFRESH_SUFFIX);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 8192, 8192, 2048, true,
                              true, true) == PackedCopyMode::COMPLETE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 513, 8192, 513, true, true,
                              true) == PackedCopyMode::COMPLETE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 8, 8192, 8, true, false,
                              true) == PackedCopyMode::COMPLETE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 8, 8192, 8, true, true,
                              false) == PackedCopyMode::COMPLETE);
  assert(UpdatePackedCopyMode(PackedCopyMode::REFRESH_SUFFIX, 32768, 8192, 2048,
                              true, true, true) == PackedCopyMode::COMPLETE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 512, 0, 512, false, false,
                              false) == PackedCopyMode::NONE);
  assert(UpdatePackedCopyMode(PackedCopyMode::NONE, 513, 0, 513, false, false,
                              false) == PackedCopyMode::COMPLETE);
}
