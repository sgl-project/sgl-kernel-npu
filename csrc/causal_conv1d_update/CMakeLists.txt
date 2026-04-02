# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_host/tiling
)

set(CAUSAL_CONV1D_UPDATE_OP_KERNEL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_kernel/causal_conv1d_update.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_kernel/causal_conv1d_update.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_kernel/causal_conv1d_update_struct.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_kernel/causal_conv1d_update_base.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../causal_conv1d_update/op_kernel/causal_conv1d_update_tilingdata.h
)
