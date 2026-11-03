// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstddef>

#include "contiguous_index_runs.h"

namespace sglang {
namespace npu_kernel {

struct KvH2dCopy {
    size_t device_offset;
    size_t host_offset;
    size_t device_pitch;
    size_t host_pitch;
    size_t width;
    size_t height;
};

// Device is [layer, page, payload], host is [page, layer, payload].
// Preserve input order across runs, including repeated destination pages.
template <class Submit>
void visit_kv_h2d_copies(const std::vector<int64_t> &device_pages, const std::vector<int64_t> &host_pages,
                         size_t layers, size_t device_page_count, size_t page_bytes, Submit submit)
{
    constexpr size_t max_height = 5 * 1024 * 1024;  // aclrtMemcpy2dAsync limit
    for (const auto &[begin, count] :
         build_contiguous_runs(device_pages.data(), host_pages.data(), static_cast<int64_t>(device_pages.size()))) {
        size_t device = static_cast<size_t>(device_pages[begin]);
        size_t host = static_cast<size_t>(host_pages[begin]);
        for (size_t left = static_cast<size_t>(count); left > 0;) {
            const size_t pages = std::min(left, max_height);
            if (pages > layers) {
                for (size_t layer = 0; layer < layers; ++layer) {
                    submit(KvH2dCopy{(layer * device_page_count + device) * page_bytes,
                                     (host * layers + layer) * page_bytes, page_bytes, layers * page_bytes, page_bytes,
                                     pages});
                }
            } else {
                for (size_t page = 0; page < pages; ++page) {
                    submit(KvH2dCopy{(device + page) * page_bytes, (host + page) * layers * page_bytes,
                                     device_page_count * page_bytes, page_bytes, page_bytes, layers});
                }
            }
            device += pages;
            host += pages;
            left -= pages;
        }
    }
}

}  // namespace npu_kernel
}  // namespace sglang
