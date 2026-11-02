#ifndef TILING_ARGS_H
#define TILING_ARGS_H

#include <cstdint>

namespace Moe {
namespace A3WindowLayout {
// Keep these host-side layout values synchronized with op_kernel/window_layout.h.
constexpr uint64_t KB = 1024UL;
constexpr uint64_t MB = 1024UL * KB;

// A3 windowsIn layout for one ping-pong half:
//
//   windowsIn + dataState * (totalWinSize / 2)
//   +---------------------------+---------------------------------------------+
//   | byte range                | owner / purpose                             |
//   +---------------------------+---------------------------------------------+
//   | [0, 102MB)                | notify-dispatch state/payload               |
//   | [102MB, 106MB)            | normal combine token-state                  |
//   | [106MB, 106MB + 24KB)     | ll dispatch selector metadata (48 * 512B)   |
//   | [106MB + 24KB, 106MB + 536KB)    | ll dispatch working state              |
//   | [106MB + 536KB, 106MB + 560KB)   | ll combine selector metadata (48*512B)|
//   | [106MB + 560KB, 107MB + 48KB)    | ll combine working state               |
//   | [107MB + 48KB, halfSize)         | unified data area for normal and ll    |
//   +---------------------------+---------------------------------------------+

constexpr uint64_t kNotifyDispatchSize = 102UL * MB;
constexpr uint64_t kNormalCombineStateSize = 4UL * MB;
constexpr uint64_t kNormalCombineStateHalfSize = kNormalCombineStateSize / 2UL;
constexpr uint64_t kNormalCombineStateEntrySize = 32UL;
constexpr uint64_t kAivCount = 48UL;
constexpr uint64_t kAivMetadataStride = 512UL;
constexpr uint64_t kLlSelectorMetadataSize = kAivCount * kAivMetadataStride;
constexpr uint64_t kLlStateTimeoutBytes = 8UL * sizeof(float);
constexpr uint64_t kLlStateSize = 512UL * KB;
// Hybrid timeout probes occupy the final 32 bytes of their owning state slot.
// Legacy keeps its original +1000KB probe address in the V2 kernels.
constexpr uint64_t kLlStateTimeoutOffset = kLlStateSize - kLlStateTimeoutBytes;
constexpr uint64_t kLlStateEntrySize = 32UL;
constexpr uint64_t kLlMaxBs = 512UL;
constexpr uint64_t kLlMaxTopK = 16UL;
constexpr uint64_t kLlMaxSharedExpertNum = 4UL;

// Keep these legacy values and this layout description synchronized with
// op_kernel/window_layout.h. Legacy windowsIn, per ping-pong half:
//
//   [0, 102MB)        notify-dispatch payload/state
//   [102MB, 106MB)    normal combine token-state
//   [106MB, halfSize) normal dispatch/combine payload
//   [0, halfSize)     V2 dispatch/combine payload (overlaps normal layout)
//
// Legacy windowsExp holds shared V2 control addresses. Dispatch state starts
// at +0KB/+500KB for dataState 0/1; combine starts at +64KB/+564KB. Selector
// metadata starts at +950KB for dispatch and +975KB for combine (48 * 512B).
// These state regions are reused by sequential V2 phases, not isolated slots.
constexpr uint64_t kLegacyNormalDataOffset = kNotifyDispatchSize + kNormalCombineStateSize;
constexpr uint64_t kLegacyV2StateHalfSize = 500UL * KB;
constexpr uint64_t kLegacyV2CombineStateOffset = 64UL * KB;
constexpr uint64_t kLegacyV2DispatchSelectorOffset = 950UL * KB;
constexpr uint64_t kLegacyV2CombineSelectorOffset = 975UL * KB;
constexpr uint64_t kLegacyLlStateTimeoutOffset = 1000UL * KB;

constexpr uint64_t kLlDispatchSelectorOffset = kNotifyDispatchSize + kNormalCombineStateSize;
constexpr uint64_t kLlDispatchStateOffset = kLlDispatchSelectorOffset + kLlSelectorMetadataSize;
constexpr uint64_t kLlCombineSelectorOffset = kLlDispatchStateOffset + kLlStateSize;
constexpr uint64_t kLlCombineStateOffset = kLlCombineSelectorOffset + kLlSelectorMetadataSize;
constexpr uint64_t kDataOffset = kLlCombineStateOffset + kLlStateSize;
constexpr uint64_t kPerHalfReservedSize = kDataOffset;

static_assert(kLlStateTimeoutOffset + kLlStateTimeoutBytes <= kLlStateSize,
              "V2 timeout probe must remain inside its state slot");
static_assert(kLlMaxBs * (kLlMaxTopK + kLlMaxSharedExpertNum) * kLlStateEntrySize <= kLlStateTimeoutOffset,
              "V2 combine state must remain inside its state slot");
}  // namespace A3WindowLayout

namespace A5WindowLayout {
// Keep these host-side values synchronized with op_kernel/window_layout.h.
constexpr uint64_t KB = 1024UL;
constexpr uint64_t MB = 1024UL * KB;

// A5 windowsIn layout for one ping-pong half in hybrid deployment. Offsets
// below are relative to GetBaseWindAddrByRankId(), after the fixed 4MB
// windowsIn MTE/state prefix:
//
//   [0, 102MB)                    normal notify-dispatch state/payload
//   [102MB, 106MB)                normal combine token-state
//   [106MB, 106MB + 36KB)         v2 dispatch selector metadata
//   [106MB + 36KB, 106MB + 548KB) v2 dispatch working state
//   [106MB + 548KB, 106MB + 584KB) v2 combine selector metadata
//   [106MB + 584KB, 107MB + 72KB) v2 combine working state
//   [107MB + 72KB, halfSize)      shared normal/v2 token data
//
// State and selector ranges are independent in hybrid mode. The common data
// area starts at kDataOffset and is repeated for dataState 0 and 1. The 4MB
// prefix is window-global, so host reserved-size calculations charge 2MB to
// each half.
constexpr uint64_t kMteStateWinSize = 4UL * MB;
constexpr uint64_t kNotifyDispatchSize = 102UL * MB;
constexpr uint64_t kNormalCombineStateSize = 4UL * MB;
constexpr uint64_t kNormalCombineStateHalfSize = kNormalCombineStateSize / 2UL;
constexpr uint64_t kNormalCombineStateEntrySize = 32UL;
constexpr uint64_t kAivCount = 72UL;
constexpr uint64_t kAivMetadataStride = 512UL;
constexpr uint64_t kLlSelectorMetadataSize = kAivCount * kAivMetadataStride;
constexpr uint64_t kLlStateTimeoutBytes = 8UL * sizeof(float);
constexpr uint64_t kLlStateSize = 512UL * KB;
constexpr uint64_t kLlStateTimeoutOffset = kLlStateSize - kLlStateTimeoutBytes;
constexpr uint64_t kLlStateEntrySize = 32UL;
constexpr uint64_t kLlMaxBs = 512UL;
constexpr uint64_t kLlMaxTopK = 16UL;
constexpr uint64_t kLlMaxSharedExpertNum = 4UL;
static_assert(kLlSelectorMetadataSize <= 50UL * KB, "A5 selector metadata must fit in the 50KB selector control area");
constexpr uint64_t kLegacyNormalDataOffset = kNotifyDispatchSize + kNormalCombineStateSize;
constexpr uint64_t kLlDispatchSelectorOffset = kLegacyNormalDataOffset;
constexpr uint64_t kLlDispatchStateOffset = kLlDispatchSelectorOffset + kLlSelectorMetadataSize;
constexpr uint64_t kLlCombineSelectorOffset = kLlDispatchStateOffset + kLlStateSize;
constexpr uint64_t kLlCombineStateOffset = kLlCombineSelectorOffset + kLlSelectorMetadataSize;
constexpr uint64_t kDataOffset = kLlCombineStateOffset + kLlStateSize;
// The A5 MTE prefix is window-global; when host tiling computes
// (perHalfDataSize + reservedSize) * 2, only half of the prefix belongs to
// each ping-pong half.
constexpr uint64_t kPerHalfReservedSize = kMteStateWinSize / 2UL + kDataOffset;
static_assert(kPerHalfReservedSize * 2UL == kMteStateWinSize + kDataOffset * 2UL,
              "A5 reserved size must count the MTE prefix exactly once");

static_assert(kLlStateTimeoutOffset + kLlStateTimeoutBytes <= kLlStateSize,
              "A5 V2 timeout probe must remain inside its state slot");
static_assert(kLlMaxBs * (kLlMaxTopK + kLlMaxSharedExpertNum) * kLlStateEntrySize <= kLlStateTimeoutOffset,
              "A5 V2 combine state must remain inside its state slot");
}  // namespace A5WindowLayout
}  // namespace Moe

#endif  // TILING_ARGS_H
