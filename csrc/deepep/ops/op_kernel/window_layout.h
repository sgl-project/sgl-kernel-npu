#ifndef WINDOW_LAYOUT_H
#define WINDOW_LAYOUT_H

#include <cstdint>

namespace Moe {
namespace A3WindowLayout {
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
//
// The state areas above are private to their owners. Normal/ll payload data
// must start at kDataOffset, so neither may overwrite low-latency notify or
// normal/ll synchronization state. The same layout is repeated in both
// ping-pong halves; dataState selects the active half.
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

// Legacy layout is safe only when normal and V2 low-latency operators are
// never deployed against the same HCCL window. It intentionally reuses the
// following areas instead of isolating the two operator families:
//
//   windowsIn, per ping-pong half:
//   +---------------------------+---------------------------------------------+
//   | byte range                | legacy user / purpose                       |
//   +---------------------------+---------------------------------------------+
//   | [0, 102MB)                | notify-dispatch payload/state               |
//   | [102MB, 106MB)            | normal combine token-state                  |
//   | [106MB, halfSize)         | normal dispatch/combine payload             |
//   | [0, halfSize)             | ll dispatch/combine payload (overlaps above)|
//   +---------------------------+---------------------------------------------+
//
//   windowsExp, V2 logical control addresses (shared between V2 phases):
//   dispatch state: dataState 0 at +0KB,   dataState 1 at +500KB
//   combine  state: dataState 0 at +64KB,  dataState 1 at +564KB
//   dispatch selector metadata: +950KB (48 * 512B)
//   combine  selector metadata: +975KB (48 * 512B)
//
// The V2 state ranges are deliberately reused by sequential dispatch/combine.
// They must not be considered independent state slots.
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
}  // namespace Moe

#endif  // WINDOW_LAYOUT_H
