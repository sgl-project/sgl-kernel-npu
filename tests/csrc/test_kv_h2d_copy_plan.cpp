// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

#include "kv_h2d_copy_plan.h"

namespace {
using sglang::npu_kernel::KvH2dCopy;
using sglang::npu_kernel::visit_kv_h2d_copies;

size_t check(const std::vector<int64_t> &dst, const std::vector<int64_t> &src, size_t layers, size_t width)
{
    constexpr size_t num_pages = 19;
    std::vector<uint8_t> host(num_pages * layers * width);
    std::mt19937 random(571);
    std::generate(host.begin(), host.end(), [&] { return static_cast<uint8_t>(random()); });
    std::vector<uint8_t> expected(host.size(), 0xab), result = expected;
    for (size_t page = 0; page < dst.size(); ++page) {
        for (size_t layer = 0; layer < layers; ++layer) {
            std::memcpy(expected.data() + (layer * num_pages + dst[page]) * width,
                        host.data() + (src[page] * layers + layer) * width, width);
        }
    }
    size_t calls = 0;
    visit_kv_h2d_copies(dst, src, layers, num_pages, width, [&](const KvH2dCopy &copy) {
        ++calls;
        for (size_t row = 0; row < copy.height; ++row) {
            const auto d = copy.device_offset + row * copy.device_pitch;
            const auto h = copy.host_offset + row * copy.host_pitch;
            if (d + copy.width > result.size() || h + copy.width > host.size()) {
                throw std::runtime_error("descriptor exceeds the tensor storage");
            }
            std::memcpy(result.data() + d, host.data() + h, copy.width);
        }
    });
    if (result != expected || calls > dst.size()) {
        throw std::runtime_error("descriptor result or count differs from the page reference");
    }
    return calls;
}
}  // namespace

int main()
{
    if (check({4, 5, 6, 7, 8}, {0, 1, 2, 3, 4}, 2, 32) != 2 ||
        check({4, 6, 5}, {0, 1, 2}, 2, 32) != 3 ||
        check({0, 1, 2}, {0, 4, 3}, 2, 32) != 3 ||
        check({}, {}, 2, 32) != 0) {
        return 1;
    }
    std::mt19937 random(541);
    for (size_t i = 0; i < 1000; ++i) {
        std::vector<int64_t> src, dst;
        for (size_t run = 0; run < 5; ++run) {
            const size_t n = 1 + random() % 9;
            const size_t s = random() % (20 - n), d = random() % (20 - n);
            for (size_t j = 0; j < n; ++j) {
                src.push_back(s + j);
                dst.push_back(d + j);
            }
        }
        check(dst, src, 1 + i % 5, 1 + i % 64);
    }
}
