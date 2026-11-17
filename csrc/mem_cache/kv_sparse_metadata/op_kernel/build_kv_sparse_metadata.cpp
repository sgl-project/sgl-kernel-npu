#include "../metadata_args.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include "cpu_kernel.h"
#include "log.h"
namespace {
const char *const kBuildKvSparseMetadata = "BuildKvSparseMetadata";

void CacheRange(const void *ptr, size_t bytes, bool invalidate, uintptr_t line_size)
{
    if (bytes == 0) return;
    const auto start = reinterpret_cast<uintptr_t>(ptr);
    for (uintptr_t p = start & ~(line_size - 1); p < start + bytes; p += line_size) {
        if (invalidate) {
            asm volatile("dc civac, %0" :: "r"(p) : "memory");
        } else {
            asm volatile("dc cvac, %0" :: "r"(p) : "memory");
        }
    }
}
void CacheBarrier()
{
    asm volatile("dsb sy" ::: "memory");
}
}
namespace aicpu {
static uint32_t BuildRecords(
    const sglang::kv_metadata::Args *args)
{
    using namespace sglang::kv_metadata;
    if (!args || !args->components || args->components > kMaxComponents ||
        !args->page_size || (args->direction != 1 && args->direction != 2) ||
        (args->pages && (!args->device_indices || !args->host_indices)) || !args->count) return 1;
    uint64_t offsets[kMaxComponents]{};
    uint64_t chunks[kMaxComponents]{};
    uint64_t records = 0;
    uint64_t split_record = kMaxRecords;
    for (uint64_t c = 0; c < args->components; ++c) {
        const auto &m = args->component[c];
        if (!m.layers ||
            !m.device_base || !m.host_base || !m.page_bytes || m.page_bytes > INT32_MAX) return 1;
        chunks[c] = ChunksPerPage(m.page_bytes);
        if (args->pages > (kMaxRecords - records) / m.layers / chunks[c]) return 1;
        offsets[c] = records;
        if (chunks[c] > 1 && split_record == kMaxRecords) split_record = records;
        records += args->pages * m.layers * chunks[c];
    }
    const uint64_t total = CopyRecordCount(records, split_record != kMaxRecords);
    auto *src = reinterpret_cast<uint64_t *>(args->src);
    auto *dst = reinterpret_cast<uint64_t *>(args->dst);
    auto *lengths = reinterpret_cast<int32_t *>(args->lengths);
    auto *count = reinterpret_cast<uint32_t *>(args->count);
    if (records && (!src || !dst || !lengths)) return 1;
    uint64_t ctr;
    asm volatile("mrs %0, ctr_el0" : "=r"(ctr));
    const uintptr_t line_size = 4ULL << ((ctr >> 16) & 15);
    CacheRange(src, total * sizeof(*src), true, line_size);
    CacheRange(dst, total * sizeof(*dst), true, line_size);
    CacheRange(lengths, total * sizeof(*lengths), true, line_size);
    CacheRange(count, sizeof(*count), true, line_size);
    const auto *di = reinterpret_cast<const int64_t *>(args->device_indices);
    const auto *hi = reinterpret_cast<const int64_t *>(args->host_indices);
    // Invalidate only the cache lines containing page starts. Batch the
    // barrier across all pages rather than stalling twice per page.
    for (uint64_t p = 0; p < args->pages; ++p) {
        CacheRange(di + p * args->page_size, sizeof(*di), true, line_size);
        CacheRange(hi + p * args->page_size, sizeof(*hi), true, line_size);
    }
    CacheBarrier();
    for (uint64_t p = 0; p < args->pages; ++p) {
        // Only page starts are read, once per page across all components.
        const int64_t dt = di[p * args->page_size], ht = hi[p * args->page_size];
        const uint64_t dp = dt >= 0 ? static_cast<uint64_t>(dt) / args->page_size : 0;
        const uint64_t hp = ht >= 0 ? static_cast<uint64_t>(ht) / args->page_size : 0;
        for (uint64_t c = 0; c < args->components; ++c) {
            const auto &m = args->component[c];
            const bool valid = dt >= 0 && ht >= 0 &&
                static_cast<uint64_t>(dt) % args->page_size == 0 &&
                static_cast<uint64_t>(ht) % args->page_size == 0 &&
                dp < m.device_pages && hp < m.host_pages;
            if (!valid) {
                // MF may already be queued. Publish an empty batch before
                // reporting the error so it cannot consume partial tables.
                *count = 0;
                CacheRange(count, sizeof(*count), false, line_size);
                CacheBarrier();
                KERNEL_LOG_ERROR(
                    "Invalid KV page start at page %llu, component %llu: device=%lld, host=%lld",
                    static_cast<unsigned long long>(p), static_cast<unsigned long long>(c),
                    static_cast<long long>(dt), static_cast<long long>(ht));
                return 1;
            }
            for (uint64_t layer = 0; layer < m.layers; ++layer) {
                const uint64_t first = offsets[c] + (p * m.layers + layer) * chunks[c];
                const uint64_t da = m.device_base + layer * m.device_layer_stride + dp * m.device_page_stride;
                const uint64_t ha = m.host_base + layer * m.host_layer_stride + hp * m.host_page_stride;
                for (uint64_t chunk = 0; chunk < chunks[c]; ++chunk) {
                    const uint64_t offset = chunk * kMaxCopyBytes;
                    const uint64_t remaining = m.page_bytes - offset;
                    src[first + chunk] = (args->direction == 1 ? ha : da) + offset;
                    dst[first + chunk] = (args->direction == 1 ? da : ha) + offset;
                    lengths[first + chunk] = static_cast<int32_t>(
                        remaining < kMaxCopyBytes ? remaining : kMaxCopyBytes);
                }
            }
        }
    }
    if (total != records) {
        // Preserve all payload bytes when the AIV paired path needs one more row.
        constexpr uint64_t half = kMaxCopyBytes / 2;
        src[records] = src[split_record] + half;
        dst[records] = dst[split_record] + half;
        lengths[records] = lengths[split_record] = half;
    }
    // AICPU caches are not assumed coherent with GM or MF's AICPU task.
    // Publish all tables before publishing count.
    CacheRange(src, total * sizeof(*src), false, line_size);
    CacheRange(dst, total * sizeof(*dst), false, line_size);
    CacheRange(lengths, total * sizeof(*lengths), false, line_size);
    CacheBarrier();
    *count = static_cast<uint32_t>(total);
    CacheRange(count, sizeof(*count), false, line_size);
    CacheBarrier();
    return 0;
}

class KvSparseMetadataKernel : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext &ctx) override
    {
        using namespace sglang::kv_metadata;
        auto *di = ctx.Input(0);
        auto *hi = ctx.Input(1);
        auto *src = ctx.Output(0);
        auto *dst = ctx.Output(1);
        auto *lengths = ctx.Output(2);
        auto *count = ctx.Output(3);
        const auto infoAttr = ctx.GetAttr("component_info");
        const auto pageAttr = ctx.GetAttr("page_size");
        const auto directionAttr = ctx.GetAttr("direction");
        if (!di || !hi || !src || !dst || !lengths || !count || !infoAttr || !pageAttr || !directionAttr) return 1;
        if (di->GetDataType() != DT_INT64 || hi->GetDataType() != DT_INT64 ||
            src->GetDataType() != DT_INT64 || dst->GetDataType() != DT_INT64 ||
            lengths->GetDataType() != DT_INT32 || count->GetDataType() != DT_INT32) return 1;
        const auto info = infoAttr->GetListInt();
        const auto pageSize = pageAttr->GetInt();
        const auto direction = directionAttr->GetInt();
        if (pageSize <= 0 || (direction != 1 && direction != 2) ||
            info.empty() || info.size() > kMaxComponents * kComponentFields || info.size() % kComponentFields ||
            di->NumElements() != hi->NumElements() || di->NumElements() < 0 ||
            di->NumElements() % pageSize || count->NumElements() != 1) return 1;
        Args args{};
        args.device_indices = reinterpret_cast<uintptr_t>(di->GetData());
        args.host_indices = reinterpret_cast<uintptr_t>(hi->GetData());
        args.src = reinterpret_cast<uintptr_t>(src->GetData());
        args.dst = reinterpret_cast<uintptr_t>(dst->GetData());
        args.lengths = reinterpret_cast<uintptr_t>(lengths->GetData());
        args.count = reinterpret_cast<uintptr_t>(count->GetData());
        args.pages = di->NumElements() / pageSize;
        args.page_size = pageSize;
        args.direction = direction;
        args.components = info.size() / kComponentFields;
        // Preserve address bits even when the signed attribute is negative.
        std::memcpy(args.component, info.data(), info.size() * sizeof(int64_t));
        uint64_t total = 0;
        bool split_pages = false;
        bool aligned_pages = true;
        for (uint64_t c = 0; c < args.components; ++c) {
            const auto &m = args.component[c];
            if (!m.layers || !m.device_pages || !m.host_pages || !m.page_bytes ||
                m.page_bytes > INT32_MAX) return 1;
            const uint64_t chunks = ChunksPerPage(m.page_bytes);
            if (args.pages > (kMaxRecords - total) / m.layers / chunks) return 1;
            for (bool host : {false, true}) {
                const auto base = host ? m.host_base : m.device_base;
                const auto pages = host ? m.host_pages : m.device_pages;
                const auto ls = host ? m.host_layer_stride : m.device_layer_stride;
                const auto ps = host ? m.host_page_stride : m.device_page_stride;
                const __uint128_t last = static_cast<__uint128_t>(base) +
                    static_cast<__uint128_t>(m.layers - 1) * ls +
                    static_cast<__uint128_t>(pages - 1) * ps + m.page_bytes;
                if (!base || last > UINT64_MAX) return 1;
            }
            total += args.pages * m.layers * chunks;
            split_pages |= chunks > 1;
            aligned_pages &= m.page_bytes % kAivCopyAlignment == 0;
        }
        if (split_pages && total >= kMinAivPairedRecords && !aligned_pages) return 1;
        total = CopyRecordCount(total, split_pages);
        if (src->NumElements() != total || dst->NumElements() != total || lengths->NumElements() != total) return 1;
        return BuildRecords(&args);
    }
};
REGISTER_CPU_KERNEL(kBuildKvSparseMetadata, KvSparseMetadataKernel);
}
