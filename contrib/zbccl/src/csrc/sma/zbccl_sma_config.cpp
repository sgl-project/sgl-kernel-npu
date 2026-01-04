/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * ZBCCL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifdef A
#include "zbccl_sma_config.h"

namespace zbccl {
namespace sma {
void SMAConfig::ParseEnv()
{
    auto strEnv = ::getenv(PYTORCH_NPU_ALLOC_CONF);
    if (strEnv == nullptr) {
        return;
    }
}
}  // namespace sma
}  // namespace zbccl
#endif
