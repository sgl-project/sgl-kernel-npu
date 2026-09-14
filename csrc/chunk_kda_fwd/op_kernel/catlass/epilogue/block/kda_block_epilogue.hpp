/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_KDA_BLOCK_EPILOGUE_HPP
#define CATLASS_KDA_BLOCK_EPILOGUE_HPP

#include "catlass/kda_catlass.hpp"

namespace Catlass::Epilogue::Block {

template <
    class DispatchPolicy,
    class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace Catlass::Epilogue::Block

// chunk_kda_fwd only instantiates BlockEpilogue for its own FwdH policies
// (EpilogueAtlasKdaFwdHVnew / EpilogueAtlasKdaFwdHUpdate), whose specializations
// live under arch22|arch35/fwd_h/epilogue/block/ and are included by their users.
// None of the upstream catlass epilogues were ever selected here.

#endif // CATLASS_KDA_BLOCK_EPILOGUE_HPP
