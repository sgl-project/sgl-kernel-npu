/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_DISPATCH_POLICY_HPP
#define CATLASS_GEMM_DISPATCH_POLICY_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"

namespace Catlass::Gemm {

// Block Mmad Policies

template <class ArchTag_, bool ASYNC_ = false>
struct MmadBase {
    using ArchTag = ArchTag_;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadBase<Arch::AtlasA2, false>;
using MmadAtlasA2Async = MmadBase<Arch::AtlasA2, true>;

// Now ENABLE_UNIT_FLAG_ must be false when input element is int8
template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingPongWithPrologue : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2SingleCoreSplitk : public MmadAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static_assert(!(ENABLE_UNIT_FLAG && (L0C_STAGES > 1)), "When L0C_STAGES > 1, can not enable unitflag");
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongSliceKWithPrologue : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Preload : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

struct MmadAtlasA2FAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAQKTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2MLAPVTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2AMLAPVTp1Spec : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsync : public MmadAtlasA2Async {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;  // Stages of emitting load instruction in advance
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsyncWithCallback
    : public MmadAtlasA2PreloadAsync<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                     ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2DynamicPreloadAsyncWithCallback
    : public MmadAtlasA2PreloadAsync<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                     ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2W4A4MatmulPerTokenPerChannelDequant
    : public MmadAtlasA2PreloadAsyncWithCallback<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                                 ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};
////////////////////
// new add
template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false, bool ENABLE_ABBA_ = false>
struct GemmAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool ENABLE_ABBA = ENABLE_ABBA_;
};

struct GemvAtlasA2 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
};
////////////////////

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongBias : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAIPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAITailQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FAITailPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2FullLoadA : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false, bool USE_HF32_MODE_ = false,
          uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 1,
          uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadFullLoadA : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 1, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadAscend950FullLoadA : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2W8A16 : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicCommon : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2DynamicSmall : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Streamk : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2DynamicStreamk : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2DynamicSingleCoreSplitk : public MmadAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static_assert(!(ENABLE_UNIT_FLAG && (L0C_STAGES > 1)), "When L0C_STAGES > 1, can not enable unitflag");
};

template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256, uint32_t STAGES_ = 2>
struct MmadAtlasA2DynamicAiv : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
};

// UB must have a capacity of UBA(mTile) + UBB(nTile) + UBC(mTile, nTile) size.
// MmadAtlasA2AivSimple is suitable for cases: mTile <= mTileUbLimits && nTile >= params.n
template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256, bool IS_TILE_M_ = true>
struct MmadAtlasA2DynamicAivSimple : public MmadAtlasA2 {
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
    static constexpr bool IS_TILE_M = IS_TILE_M_;
};

// mTile and nTile must be aligned to 16.
// UB must have a capacity of UBA(mTile) + UBB(nTile) + 2 * UBC(mTile, nTile) size.
// MmadAtlasA2AivTrans is suitable for cases: M > N
template <uint32_t SCALAR_BUFFER_ELE_NUM_ = 256>
struct MmadAtlasA2DynamicAivTrans : public MmadAtlasA2 {
    static constexpr uint32_t SCALAR_BUFFER_ELE_NUM = SCALAR_BUFFER_ELE_NUM_;
};

template <uint32_t STAGES_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Small : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};
// Now ENABLE_UNIT_FLAG_ must be false when input element is int8
template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadPingpong : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadPingpongMutex : public MmadBase<ArchTag_, false> {
    static_assert(std::is_same_v<ArchTag_, Arch::Ascend950>, "MmadPingpongMutex only supports Arch::Ascend950");
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 2>
struct MmadMultiBatch : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
};

template <class ArchTag_, uint32_t PRELOAD_STAGES_, uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_,
          uint32_t L0B_STAGES_, uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_,
          bool USE_HF32_MODE_ = false, bool ENABLE_L1_RESIDENT_ = false>
struct MmadPreloadAsyncWithCallback : public MmadBase<ArchTag_, true> {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

// 基于TLA提供block层不感知尾块逻辑，tile层感知originShape的流程。使用TileView与MakeTensorLike。
template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadPingpongTlaV2 : public MmadBase<ArchTag_, ENABLE_UNIT_FLAG_> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct SparseMatmulMultiBlockOnKAxis : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAITailQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIQKMx : public MmadFAIQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_> {};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAITailPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadFAIPVMx : public MmadFAIPV<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_> {};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L1_SCALE_FACTOR_K_ = 16, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadMx : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
    /// GM→L1 MX scale 一次驻留的 L1 K 条带个数。设为 1 时每条 L1 K 条带各搬一次 scale。
    static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false,
          uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadA8W4Mx : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, uint32_t L1B_STAGES_ = 1>
struct MxA8W4Prologue : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadDequant : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false>
struct MmadPingpongPertile : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false,
          uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadSvd1 : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
};
template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L0C_STAGES_ = 1, bool ENABLE_L1_RESIDENT_ = false,
          uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadSvd2 : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L1_SCALE_FACTOR_K_ = 8, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadSvd3 : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool NZ_LAYOUT_FLAG_ = false, bool PA_BNNBSD_FLAG_ = 0,
          bool ENABLE_DN_ = false>
struct MmadFlashAttentionQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t PA_BNNBSD_FLAG = PA_BNNBSD_FLAG_;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool NZ_LAYOUT_FLAG = NZ_LAYOUT_FLAG_;
    static constexpr bool ENABLE_DN = ENABLE_DN_;
};

template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool NZ_LAYOUT_FLAG_ = false, bool PA_BNNBSD_FLAG_ = 0,
          bool ENABLE_DN_ = false>
struct MmadFlashAttentionPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t PA_BNNBSD_FLAG = PA_BNNBSD_FLAG_;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool NZ_LAYOUT_FLAG = NZ_LAYOUT_FLAG_;
    static constexpr bool ENABLE_DN = ENABLE_DN_;
};

}  // namespace Catlass::Gemm

#endif  // CATLASS_GEMM_DISPATCH_POLICY_HPP
