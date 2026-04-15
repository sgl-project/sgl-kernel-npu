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
 
#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)
#include <ATen/Tensor.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <dlfcn.h>

#include <fstream>

#include "find_op_path.h"

std::string RealPath(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return "";
    }
    char realPathBuf[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPathBuf) == nullptr) {
        return "";
    }
    return std::string(realPathBuf);
}

std::vector<std::string> SplitStr(std::string s, const std::string &del)
{
    int end = s.find(del);
    std::vector<std::string> path_list;
    while (end != -1) {
        path_list.push_back(s.substr(0, end));
        s.erase(s.begin(), s.begin() + end + 1);
        end = s.find(del);
    }
    path_list.push_back(s);
    return path_list;
}
std::vector<std::string> ProcessPathList(const std::string& pathStr)
{
    return SplitStr(pathStr, ":");
}

void AppendLibPathSuffix(std::vector<std::string>& pathList)
{
    for (auto& currentPathIt : pathList) {
        currentPathIt += "/op_api/lib/";
    }
}

std::vector<std::string> ProcessCustomLibPath(const char* ascendCustomOppPath)
{
    std::string ascendCustomOppPathStr(ascendCustomOppPath);
    auto customLibPathList = ProcessPathList(ascendCustomOppPathStr);
    if (customLibPathList.empty()) {
        return std::vector<std::string>();
    }
    AppendLibPathSuffix(customLibPathList);
    return customLibPathList;
}

std::vector<std::string> GetCustomLibPath()
{
    const char *ascendCustomOppPath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if (ascendCustomOppPath == nullptr) {
        ASCEND_LOGW("ASCEND_CUSTOM_OPP_PATH is not exists");
        return std::vector<std::string>();
    }
    return ProcessCustomLibPath(ascendCustomOppPath);
}

std::string GetVendorsConfigFilePath(const std::string& vendorsPath)
{
    return RealPath(vendorsPath + "/config.ini");
}

bool IsFileExist(const std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
}

bool ValidateVendorsConfigFile(const std::string& configFile)
{
    if (configFile.empty() || !IsFileExist(configFile)) {
        ASCEND_LOGW("config.ini is not exists or the path length is more than %d", PATH_MAX);
        return false;
    }
    return true;
}

std::string ReadLoadPriorityLine(const std::string& configFile)
{
    std::ifstream ifs(configFile);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("load_priority=") == 0) {
            break;
        }
    }
    return line;
}

std::string ExtractLoadPriorityValue(const std::string& line)
{
    std::string head = "load_priority=";
    std::string result = line;
    if (result.find(head) == 0) {
        result.erase(0, head.length());
    }
    return result;
}

std::vector<std::string> ProcessVendorsList(const std::string& vendorsPath, const std::string& line)
{
    auto defaultVendorsList = SplitStr(line, ",");
    for (auto &it : defaultVendorsList) {
        it = RealPath(vendorsPath + "/" + it + "/op_api/lib/");
    }
    return defaultVendorsList;
}

std::vector<std::string> ParseVendorsConfig(const std::string& vendorsPath)
{
    std::string vendorsConfigFile = GetVendorsConfigFilePath(vendorsPath);
    if (!ValidateVendorsConfigFile(vendorsConfigFile)) {
        return {};
    }
    std::string line = ReadLoadPriorityLine(vendorsConfigFile);
    std::string priorityValue = ExtractLoadPriorityValue(line);
    return ProcessVendorsList(vendorsPath, priorityValue);
}

std::vector<std::string> GetDefaultCustomLibPath()
{
    const char *ascendOppPath = std::getenv("ASCEND_OPP_PATH");
    std::vector<std::string> defaultVendorsList;
    if (ascendOppPath == nullptr) {
        ASCEND_LOGW("ASCEND_OPP_PATH is not exists");
        return std::vector<std::string>();
    }
    std::string vendorsPath(ascendOppPath);
    vendorsPath = vendorsPath + "/vendors";
    return ParseVendorsConfig(vendorsPath);
}

const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

std::string GetCustomOpApiLibPath(const std::string& libPath)
{
    return RealPath(libPath + "/" + GetCustOpApiLibName());
}

void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

template<typename T = void>
void *GetOpApiFuncAddrInLib(T *handler, const char *libName, const std::string& apiName)
{
    auto funcAddr = dlsym(handler, apiName.c_str());
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

void* GetFuncFromDefaultLib(const std::string& apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

void* LoadDefaultCustomOpApiHandler(const std::string& defaultCustOpApiLib)
{
    if (defaultCustOpApiLib.empty()) {
        return nullptr;
    }
    return GetOpApiLibHandler(defaultCustOpApiLib.c_str());
}

void* LoadCustomOpApiHandler(const std::string& custOpApiLib)
{
    if (custOpApiLib.empty()) {
        return nullptr;
    }
    return GetOpApiLibHandler(custOpApiLib.c_str());
}

void* FindFuncInCustomLibPath(const char* apiName, const std::string& libPath)
{
    auto custOpApiLib = GetCustomOpApiLibPath(libPath);
    auto custOpApiHandler = LoadCustomOpApiHandler(custOpApiLib);
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            ASCEND_LOGI("%s is found in %s.", apiName, custOpApiLib.c_str());
            return funcAddr;
        }
    }
    return nullptr;
}

std::string GetDefaultCustomOpApiLibPath(const std::string& libPath)
{
    return RealPath(libPath + "/" + GetCustOpApiLibName());
}

void* FindFuncInDefaultLibPath(const char* apiName, const std::string& libPath)
{
    auto defaultCustOpApiLib = GetDefaultCustomOpApiLibPath(libPath);
    auto custOpApiHandler = LoadDefaultCustomOpApiHandler(defaultCustOpApiLib);
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            ASCEND_LOGI("%s is found in %s.", apiName, defaultCustOpApiLib.c_str());
            return funcAddr;
        }
    }
    return nullptr;
}

void* AllocateWorkspace(uint64_t workspaceSize, at::Tensor& workspaceTensor)
{
    if (workspaceSize == 0) {
        return nullptr;
    }

    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, options.dtype(c10::kByte));
    return const_cast<void*>(workspaceTensor.storage().data());
}

const std::vector<std::string> g_customLibPath = GetCustomLibPath();
const std::vector<std::string> g_defaultCustomLibPath = GetDefaultCustomLibPath();