/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_KDA_FWD_H_EPILOGUE_POLICIES_HPP
#define CATLASS_EPILOGUE_KDA_FWD_H_EPILOGUE_POLICIES_HPP

#include "catlass/kda_catlass.hpp"

namespace Catlass::Epilogue {

struct EpilogueAtlasKdaFwdHVnew {
    using ArchTag = Arch::AtlasA2;
};

struct EpilogueAtlasKdaFwdHUpdate {
    using ArchTag = Arch::AtlasA2;
};

} // namespace Catlass::Epilogue

#endif // CATLASS_EPILOGUE_KDA_FWD_H_EPILOGUE_POLICIES_HPP
