#include "../csrc/fused_quant_scatter_copy_sparse_flash_attention/op_kernel/fused_quant_copy_sfa_tiling_data.h"
#include "attention/fused_quant_scatter_copy_sparse_flash_attention/op_host/fused_quant_scatter_copy_sparse_flash_attention_tiling.h"
// SPDX-License-Identifier: Apache-2.0
// Compare the direct-launch ABI with the original CANN serializer on CPU.
#include <cstdio>
#include <cstring>
int main() {
  optiling::FusedQuantScatterCopySparseFlashAttentionTilingData original;
  original.baseParams.set_batchSize(1);
  original.baseParams.set_seqSize(2);
  original.baseParams.set_qSeqSize(3);
  original.baseParams.set_nNumOfQInOneGroup(4);
  original.baseParams.set_sparseBlockCount(5);
  original.baseParams.set_sparseMode(6);
  original.baseParams.set_outputLayout(7);
  original.baseParams.set_blockSize(8);
  original.baseParams.set_maxBlockNumPerBatch(9);
  original.baseParams.set_isActualLenDimsNull(10);
  original.baseParams.set_isActualLenDimsKVNull(11);
  original.baseParams.set_dSizeVInput(12);
  original.baseParams.set_scaleValue(13.25f);
  original.singleCoreParams.set_usedCoreNum(14);
  original.set_dramMaxBlocks(15);
  original.set_hbmPhysicalBlocks(16);
  original.set_dramPhysicalBlocks(17);
  unsigned char bytes[256]{};
  original.SaveToBuffer(bytes, sizeof(bytes));
  FusedQuantScatterCopySparseFlashAttentionTilingData port{
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.25f}, {14}, 15, 16, 17, {0}};
  std::printf("original_size=%zu port_size=%zu\n", original.GetDataSize(),
              sizeof(port));
  for (unsigned i = 0; i < original.GetDataSize(); i += 4) {
    unsigned word;
    std::memcpy(&word, bytes + i, 4);
    std::printf("%u: %u\n", i, word);
  }
  std::printf("equal=%d\n", original.GetDataSize() == sizeof(port) &&
                                std::memcmp(bytes, &port, sizeof(port)) == 0);
  return original.GetDataSize() == sizeof(port) &&
                 std::memcmp(bytes, &port, sizeof(port)) == 0
             ? 0
             : 1;
}
