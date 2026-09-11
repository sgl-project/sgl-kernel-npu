









/*!
 * \file kv_quant_sparse_attn_sharedkv_common.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_attn_sharedkv_metadata.h"

using namespace AscendC;

enum class SAS_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

enum class SASTemplateMode {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H
