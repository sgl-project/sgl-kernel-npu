/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef FIND_FUNC_IN_CUSTOM_LIB_PATH_H
#define FIND_FUNC_IN_CUSTOM_LIB_PATH_H

#include <string>
#include <vector>

void* FindFuncInCustomLibPath(const char* apiName, const std::string& libPath);
const char *GetOpApiLibName(void);
void* GetFuncFromDefaultLib(const std::string& apiName);
void* FindFuncInDefaultLibPath(const char* apiName, const std::string& libPath);
void* AllocateWorkspace(uint64_t workspaceSize, at::Tensor& workspaceTensor);
extern const std::vector<std::string> g_customLibPath;
extern const std::vector<std::string> g_defaultCustomLibPath;
#endif // FIND_FUNC_IN_CUSTOM_LIB_PATH_H