#pragma once

#include <iostream>

#include "shmem_api.h"
#include "config.hpp"
#include "exception.hpp"

#define NUM_BUFFER_ALIGNMENT_BYTES 128

namespace deep_ep {
namespace internode {

int init(int rank, int num_ranks, uint64_t localMemSize, const char *server_ipport)
{
    std::cout << "rank: " << rank << " num_ranks: " << num_ranks << " localMemSize: " << localMemSize << " init start~"
              << std::endl;
    int32_t status = 0;
    shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank, num_ranks, localMemSize, server_ipport, &attributes);
    EP_HOST_ASSERT(status == SHMEM_SUCCESS);
    status = shmem_init_attr(attributes);
    EP_HOST_ASSERT(status == SHMEM_SUCCESS);
    EP_HOST_ASSERT(shmem_init_status() == SHMEM_STATUS_IS_INITIALIZED);
    std::cout << "rank: " << rank << " num_ranks: " << num_ranks << " init done!" << std::endl;

    return shmem_my_pe();
}

void *alloc(size_t size, size_t alignment)
{
    return shmem_align(alignment, size);
}

void free(void *ptr)
{
    shmem_free(ptr);
}

// void barrier() {
//     shmem_barrier_all();
// }

void finalize()
{
    EP_HOST_ASSERT(shmem_finalize() == SHMEM_SUCCESS);
}
}  // namespace internode

}  // namespace deep_ep
