// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace sglang {
namespace npu_kernel {

inline std::vector<std::pair<int64_t, int64_t>> build_contiguous_runs(
    const int64_t *device_indices, const int64_t *host_indices, int64_t count)
{
    std::vector<std::pair<int64_t, int64_t>> runs;
    runs.reserve(count);
    for (int64_t begin = 0; begin < count;) {
        int64_t end = begin + 1;
        while (end < count && device_indices[end] == device_indices[end - 1] + 1 &&
               host_indices[end] == host_indices[end - 1] + 1) {
            ++end;
        }
        runs.emplace_back(begin, end - begin);
        begin = end;
    }
    return runs;
}

}  // namespace npu_kernel
}  // namespace sglang
