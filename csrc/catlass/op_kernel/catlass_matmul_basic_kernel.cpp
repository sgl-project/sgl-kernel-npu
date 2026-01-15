// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* include file of ascendc */
#include "kernel_operator.h"
#include "../op_host/catlass_matmul_tiling.h"
/* include file of catlass */
#include "catlass/gemm/kernel/basic_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

using namespace Catlass;

extern "C" __global__ __aicore__ void catlass_matmul_basic(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmWorkspace,
                                                           GM_ADDR gmTiling)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    // tile shape for different element size
    using L1TileShape_2B = GemmShape<128, 256, 256>;
    using L0TileShape_2B = GemmShape<128, 256, 64>;
    using L1TileShape_4B = GemmShape<128, 128, 256>;
    using L0TileShape_4B = GemmShape<128, 128, 64>;
    using BlockEpilogue = void;
    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    /* init catlass template 1. fp16 no_weight_nz */
    using BlockMmad_case1 =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape_2B, L0TileShape_2B, Gemm::GemmType<half, layout::RowMajor>,
                               Gemm::GemmType<half, layout::RowMajor>, Gemm::GemmType<half, layout::RowMajor>>;
    using MatmulKernel_fp16_no_nz = Gemm::Kernel::BasicMatmul<BlockMmad_case1, BlockEpilogue, BlockScheduler>;
    /* init catlass template 2. bf16 no_weight_nz */
    using BlockMmad_case2 =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape_2B, L0TileShape_2B, Gemm::GemmType<__bf16, layout::RowMajor>,
                               Gemm::GemmType<__bf16, layout::RowMajor>, Gemm::GemmType<__bf16, layout::RowMajor>>;
    using MatmulKernel_bf16_no_nz = Gemm::Kernel::BasicMatmul<BlockMmad_case2, BlockEpilogue, BlockScheduler>;
    /* init catlass template 3. fp32 no_weight_nz */
    using BlockMmad_case3 =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape_4B, L0TileShape_4B, Gemm::GemmType<float, layout::RowMajor>,
                               Gemm::GemmType<float, layout::RowMajor>, Gemm::GemmType<float, layout::RowMajor>>;
    using MatmulKernel_fp32_no_nz = Gemm::Kernel::BasicMatmul<BlockMmad_case3, BlockEpilogue, BlockScheduler>;

    auto tiling_data = reinterpret_cast<__gm__ sglang::npu_kernel::KernelCatlassMatmulTilingData *>(gmTiling);
    uint32_t m = tiling_data->m;
    uint32_t n = tiling_data->n;
    uint32_t k = tiling_data->k;

    layout::RowMajor layoutA{m, k};
    layout::RowMajor layoutB{k, n};
    layout::RowMajor layoutC{m, n};

    /* init catlass instance and run */
    GemmCoord problemShape{m, n, k};
    if (tiling_data->data_format_mode == sglang::npu_kernel::DataFormatMode::BF16) {
        MatmulKernel_bf16_no_nz::Arguments arguments{problemShape, gmA, gmB, gmC};

        typename MatmulKernel_bf16_no_nz::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        MatmulKernel_bf16_no_nz matmul_kernel;
        matmul_kernel(params);
    } else if (tiling_data->data_format_mode == sglang::npu_kernel::DataFormatMode::FP16) {
        MatmulKernel_fp16_no_nz::Arguments arguments{problemShape, gmA, gmB, gmC};

        typename MatmulKernel_fp16_no_nz::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        MatmulKernel_fp16_no_nz matmul_kernel;
        matmul_kernel(params);
    } else if (tiling_data->data_format_mode == sglang::npu_kernel::DataFormatMode::FP32) {
        MatmulKernel_fp32_no_nz::Arguments arguments{problemShape, gmA, gmB, gmC};

        typename MatmulKernel_fp32_no_nz::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        MatmulKernel_fp32_no_nz matmul_kernel;
        matmul_kernel(params);
    } else {
        // TODO: use device error check process
        return;
    }
}
