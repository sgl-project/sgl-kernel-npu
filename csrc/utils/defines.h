// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// The code snippet comes from [CANN].
//
// Copyright (c) [2025] [CANN]. All rights reserved.
//
// This file contains code from [CANN], which is released under
// the CANN Open Software License Agreement Version 2.0 (the "License")
// See the LICENSE file in the root directory of this source tree
// or at https://gitcode.com/cann/ops-nn/blob/master/LICENSE for details.

#ifndef SGL_KERNEL_NPU_DEFINES_H
#define SGL_KERNEL_NPU_DEFINES_H

#include "version.h"

namespace sglang {
namespace npu_kernel {

#define HOST_API __attribute__((visibility("default")))

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_DEFINES_H
