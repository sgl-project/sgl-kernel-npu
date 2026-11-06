// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Torch adapter for the kv_exchange operator domain. The schema and the
// PrivateUse1 implementations self-register here, following the independent
// operator layout; nothing about this operator is added to the centralized
// csrc/pytorch_extensions.cpp.
//
// `kv_exchange` is the production entry: it receives the split-KV
// component views plus their MF DVAs and packs the whole transfer
// description into the kernel launch parameter block (replacing the
// memfabric_hybrid offload.kv_exchange_copy call that used to build the
// meta in Python, and the later pinned-staging H2D meta upload).

#include "op_host/kv_exchange.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "kv_exchange("
        "Tensor device_indices, Tensor host_indices, "
        "Tensor[] device_components, Tensor[] host_components, "
        "int[] host_dvas, int page_size, int direction, int host_layout) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_exchange", TORCH_FN(sglang::npu_kernel::kv_exchange_submit));
}
