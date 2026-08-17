// Copyright (c) 2020, Huawei Technologies Co., Ltd
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "NPUBridge.h"

namespace sglang::npu_kernel::aclnn_adapter {
NPUStorageImpl *NPUBridge::GetNpuStorageImpl(c10::StorageImpl *storageImpl)
{
    return static_cast<NPUStorageImpl *>(storageImpl);
}

NPUStorageImpl *NPUBridge::GetNpuStorageImpl(c10::Storage &&storage)
{
    return static_cast<NPUStorageImpl *>(storage.unsafeGetStorageImpl());
}

NPUStorageImpl *NPUBridge::GetNpuStorageImpl(const at::Tensor &tensor)
{
    return static_cast<NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl());
}

NPUStorageDesc &NPUBridge::GetNpuStorageImplDesc(const at::Tensor &tensor)
{
    return static_cast<NPUStorageImpl *>(tensor.storage().unsafeGetStorageImpl())->npu_desc_;
}
}  // namespace sglang::npu_kernel::aclnn_adapter
