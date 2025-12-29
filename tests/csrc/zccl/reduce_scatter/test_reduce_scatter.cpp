/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "fp16_t.h"
// #include "bfloat16.h"
#include "utils.h"

using fp16_t = op::fp16_t;
// using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem_api.h"
#include "zccl.h"

using namespace sglang::zccl;

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *data_type;
bool zero_buff = false;

template<class T>
int test_shmem_reduce_scatter(int rank_id, int n_ranks, uint64_t local_mem_size, bool zero_buff)
{
    // 初始化ACL和SHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    ACL_CHECK(aclrtCreateStream(&stream));

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    status = shmem_init_attr(attributes);

    // Prepare FFTS address
    uint64_t fftsAddr = shmemx_get_ffts_config();

    auto env_PERF_TIMES = getEnvVar("PERF_TIMES");
    int PERF_TIMES = env_PERF_TIMES.empty() ? 1 : std::stoi(env_PERF_TIMES);

    auto env_case_num = getEnvVar("CASE_NUM");
    int case_num = env_case_num.empty() ? 1 : std::stoi(env_case_num);
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 16 * (1 << i);
        test_cases.push_back(data_len);
    }

    uint32_t reduceOp = 0;
    ZCCLDataType dataType = ZCCLDataType::ZCCL_DATA_TYPE_FP32;
    int teamId = 0;
    std::string cwd = getEnvVar("PWD");

    for (int i = 0; i < test_cases.size(); i++) {
        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];

        void *input_ptr;
        if (zero_buff) {
            input_ptr = shmem_malloc(trans_size * sizeof(T));
        } else {
            aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        }
        uint8_t *input_host;
        aclrtMallocHost(reinterpret_cast<void**>(&input_host), trans_size * sizeof(T));
        std::string inputFile = cwd + "/golden/reduce_scatter_" + std::to_string(trans_size) + "_" +
                                std::to_string(n_ranks) + "/input_gm_" + std::to_string(rank_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        size_t outSingleSize = trans_size * sizeof(T) / n_ranks;
        if (zero_buff) {
            output_ptr = shmem_malloc(outSingleSize);
        } else {
            aclrtMalloc(&output_ptr, outSingleSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        aclrtMemset(output_ptr, outSingleSize, 0, outSingleSize);

        // ReduceScatter
        for (int zz = 0; zz < PERF_TIMES; zz++) {
            if (zero_buff) {
                ZcclReduceScatterZeroBuff((uint8_t *)input_ptr, (uint8_t *)output_ptr, trans_size,
                    dataType, teamId, stream);
            } else {
                ZcclReduceScatter((uint8_t *)input_ptr, (uint8_t *)output_ptr, trans_size, 
                    dataType, teamId, stream);
            }
        }
        status = aclrtSynchronizeStream(stream);

        // Result Check
        T *output_host;
        size_t output_size = outSingleSize;
        status = aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);
        status = aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);
        uint32_t loop_time = trans_size / n_ranks;
        for (auto i = 0; i < loop_time; i++) {
            auto env_debug = getEnvVar("DEBUG");
            bool isDebug = env_debug.empty() ? false : env_debug == "1";
            if (isDebug) {
                std::cout << "rank_id=" << rank_id << ", output_host=" << static_cast<float>(output_host[i]) << "\n";
            }
        }

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void**>(&golden_host), output_size);
        std::string goldenFile = cwd + "/golden/reduce_scatter_" +
            std::to_string(trans_size) + "_" + std::to_string(n_ranks) + "/golden_" + std::to_string(rank_id) + ".bin";
        ReadFile(goldenFile, golden_host, output_size);
        for (int zz = 0; zz < trans_size / n_ranks; zz++) {
            if (!fpEquals(static_cast<float>(output_host[zz]), static_cast<float>(golden_host[zz]))) {
                std::cout << static_cast<float>(output_host[zz]) << " != " << static_cast<float>(golden_host[zz])
                          << ", trans_size is : " << trans_size << ", idx is: " << zz
                          << ", rank_id is: "<< rank_id << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // 去初始化
        status = aclrtFreeHost(input_host);
        status = aclrtFreeHost(output_host);
        status = aclrtFreeHost(golden_host);

        if (zero_buff) {
            shmem_free(input_ptr);
            shmem_free(output_ptr);
        } else {
            aclrtFree(input_ptr);
            aclrtFree(output_ptr);
        }

        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finised !! Result Correct !!" << std::endl;
        }
    }

    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    ipport = argv[3];
    g_npus = atoi(argv[4]);
    f_rank = atoi(argv[5]);
    f_npu = atoi(argv[6]);
    data_type = argv[7];
    zero_buff = (argv[8] != nullptr && std::string(argv[8]) == "1");
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);
    std::cout << "init shmem tls result:" << ret << std::endl;
    if (std::string(data_type) == "int") {
        status = test_shmem_reduce_scatter<int>(rank_id, n_ranks, local_mem_size, zero_buff);
    } else if (std::string(data_type) == "float") {
        status = test_shmem_reduce_scatter<float>(rank_id, n_ranks, local_mem_size, zero_buff);
    } else if (std::string(data_type) == "float16_t") {
        status = test_shmem_reduce_scatter<fp16_t>(rank_id, n_ranks, local_mem_size, zero_buff);
    }
    
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}