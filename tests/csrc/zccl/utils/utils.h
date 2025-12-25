/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UTILS_H
#define UTILS_H

#include <acl/acl.h>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    // lock
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // move ptr to specified offset
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // write data
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << strerror(errno) << std::endl;
    }

    // unlock
    flock(fd, LOCK_UN);

    close(fd);
    return true;
}

std::string getEnvVar(const std::string &key)
{
    const char* val = std::getenv(key.c_str());
    return val ? std::string(val) : "";
}

template<typename T>
bool fpEquals(T a, T b, T atol = 1e-3, T rtol = 1e-3)
{
    T abs = std::abs(a - b);
    if (abs <= atol) {
        return true;
    }
    return abs <= rtol * std::max(std::abs(a), std::abs(b));
}

#endif // UTILS_H