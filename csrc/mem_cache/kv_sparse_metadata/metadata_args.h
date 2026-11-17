#pragma once
#include <cstdint>
#include <type_traits>
namespace sglang::kv_metadata {
constexpr uint64_t kMaxComponents = 4;
constexpr uint64_t kMaxRecords = 65536;
// MF's A3 AIV fast path packs two records into one 88 KiB UB slot.
constexpr uint64_t kMaxCopyBytes = 32 * 1024;
constexpr uint64_t kAivCopyAlignment = 32;
constexpr uint64_t kMinAivPairedRecords = 48;

inline uint64_t ChunksPerPage(uint64_t bytes)
{
    return bytes / kMaxCopyBytes + (bytes % kMaxCopyBytes != 0);
}

inline uint64_t CopyRecordCount(uint64_t records, bool split_pages)
{
    // The AIV paired path ignores an odd final row. Split an existing full
    // chunk in two so every emitted record carries payload bytes.
    return records + (split_pages && records >= kMinAivPairedRecords ? records % 2 : 0);
}
// Host bases are MF HVAs, never AIV DVAs. Fields travel as a task attribute.
struct Component {
    uint64_t device_base, host_base;
    uint64_t device_pages, host_pages, layers;
    uint64_t device_layer_stride, device_page_stride;
    uint64_t host_layer_stride, host_page_stride, page_bytes;
};
constexpr uint64_t kComponentFields = 10;
struct Args {
    uint64_t device_indices, host_indices, src, dst, lengths, count;
    uint64_t pages, page_size, direction, components;
    Component component[kMaxComponents];
};
static_assert(sizeof(Component) == kComponentFields * sizeof(int64_t));
static_assert(std::is_trivially_copyable_v<Component>);
}
