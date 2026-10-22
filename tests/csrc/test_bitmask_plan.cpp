#include "apply_token_bitmask_plan.h"
#include "fixtures/apply_token_bitmask/legacy_plan.h"

#include <iostream>

static bool Equal(const sgl::BitmaskPlan& a, const sgl::BitmaskPlan& b)
{
    return a.numRows == b.numRows && a.vocabSize == b.vocabSize && a.logitsStride == b.logitsStride &&
           a.bitmaskStride == b.bitmaskStride && a.baseRows == b.baseRows && a.extraCores == b.extraCores &&
           a.tileLength == b.tileLength && a.blockDim == b.blockDim && a.dtypeSize == b.dtypeSize;
}

int main()
{
    // Synthetic arithmetic fixtures, not measurements or settings for the target NPU.
    // Widths include existing public test cases; padding follows the existing caller.
    const int64_t rows[] = {0, 1, 2, 4, 8, 16, 32, 65};
    const int64_t widths[] = {1, 31, 32, 33, 64, 100, 128, 320, 32000, 32100, 128256};
    const int64_t cores[] = {0, 1, 3, 8, 32};
    const uint64_t capacities[] = {0, 16384, 16385, 32768, 196608};
    size_t cases = 0;
    for (auto n : rows) for (auto width : widths) for (auto core : cores)
    for (auto ub : capacities) for (int64_t bytes : {2, 4}) for (int64_t extraMask : {0, 5}) {
        const int64_t padded = (width + 255) / 256 * 256;
        const int64_t stride = padded / 32 + extraMask;
        const auto actual = sgl::MakeBitmaskPlan(n, padded, stride, bytes, core, ub);
        const auto baseline = LegacyPlan(n, padded, stride, bytes, core, ub);
        if (!Equal(actual, baseline) || actual.baseRows * actual.blockDim + actual.extraCores != n ||
            actual.extraCores >= actual.blockDim || actual.tileLength % 256 != 0 ||
            actual.tileLength < 256 || actual.tileLength > padded || actual.bitmaskStride != stride) {
            std::cerr << "FAIL: rows=" << n << " width=" << width << " core=" << core
                      << " ub=" << ub << " bytes=" << bytes << " mask_stride=" << stride << '\n';
            return 1;
        }
        ++cases;
    }
    const auto known = sgl::MakeBitmaskPlan(8, 32000, 1005, 4, 3, 32768);
    if (known.blockDim != 3 || known.baseRows != 2 || known.extraCores != 2 || known.tileLength != 768) {
        std::cerr << "FAIL: independently calculated scheduling example\n";
        return 1;
    }
    std::cout << "PASS: " << cases << " scheduling comparisons against the original Host formula\n";
}
