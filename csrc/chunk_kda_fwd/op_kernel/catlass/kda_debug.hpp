/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_KDA_DEBUG_HPP
#define CATLASS_KDA_DEBUG_HPP

#include <functional>
#include <iostream>
#include <sstream>

#include <acl/acl.h>

#define SINGLE_CORE_DUMPSIZE (1024 * 1024)
#ifdef CATLASS_ARCH
#if CATLASS_ARCH == 2201
#define DUMP_CORE_NUM 75
#endif
#if CATLASS_ARCH == 3510
#define DUMP_CORE_NUM 108
#endif
#endif
#ifndef DUMP_CORE_NUM
#define DUMP_CORE_NUM 0
#endif
// DUMP_CORE_NUM is from AscendC host stub
#define ALL_DUMPSIZE (DUMP_CORE_NUM * SINGLE_CORE_DUMPSIZE)

using LogFuncType = std::function<void(const char *)>;
/**
 * @brief Check acl api status code.
 * @param status The return code of acl api.
 * @param logFunc Log function, which receives a C-Style string.
 * @return
 */
inline void aclCheck(
    aclError status,
    LogFuncType logFunc = [](const char *logStrPtr) { std::cerr << logStrPtr; })
{
    if (status != ACL_SUCCESS) {
        std::stringstream ss;
        ss << "AclError: " << status;
        logFunc(ss.str().c_str());
    }
}
/**
 * @brief Check rt api status code.
 * @param status The return code of rt api.
 * @param logFunc Log function, which receives a C-Style string.
 * @return
 */
inline void rtCheck(
    int status,
    LogFuncType logFunc = [](const char *logStrPtr) { std::cerr << logStrPtr; })
{
    if (status != 0) {
        std::stringstream ss;
        ss << "RtError: " << status;
        logFunc(ss.str().c_str());
    }
}

namespace Adx {
void AdumpPrintWorkSpace(const void *dumpBufferAddr, const size_t dumpBufferSize, aclrtStream stream, const char *opType);
} // namespace Adx

#endif // CATLASS_KDA_DEBUG_HPP
