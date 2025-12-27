#pragma once

#define EXPORT_API __attribute__((visibility("default")))

#include <c10/util/Logging.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <sys/types.h>
#include <torch/extension.h>

#include <atomic>
#include <iostream>
#include <mutex>

#include "shmem_alloc_mm_heap.h"
#include "shmem_api.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUBlockHandle.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

using c10_npu::NPUCachingAllocator::BlockInfo;
using c10_npu::NPUCachingAllocator::DeviceStats;
using c10_npu::NPUCachingAllocator::RecordContext;
using c10_npu::NPUCachingAllocator::SegmentInfo;
using c10_npu::NPUCachingAllocator::Stat;
using c10_npu::NPUCachingAllocator::StatArray;
using c10_npu::NPUCachingAllocator::StatType;
using c10_npu::NPUCachingAllocator::TraceEntry;

using stream_set = ska::flat_hash_set<c10_npu::NPUStream>;

#define NPU_CHECK_SUPPORT_OR_ERROR(err_code, ...)                         \
  do {                                                                    \
    auto Error = err_code;                                                \
    static c10_npu::acl::AclErrorCode err_map;                            \
    if ((Error) != ACL_ERROR_NONE) {                                      \
      if ((Error) == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {                  \
        static auto feature_not_support_warn_once = []() {                \
          printf("[WARN]%s,%s:%u:%s\n", __FUNCTION__, __FILE__, __LINE__, \
                 "Feature is not supportted and the possible cause is"    \
                 " that driver and firmware packages do not match.");     \
          return true;                                                    \
        }();                                                              \
      } else {                                                            \
        TORCH_CHECK(false, __func__, ":", __FILE__, ":", __LINE__, "\n",  \
                    c10_npu::c10_npu_get_error_message());                \
      }                                                                   \
    }                                                                     \
  } while (0)

typedef std::shared_ptr<c10::GatheredContext> (*CreateContextFn)(void);

constexpr size_t kMinBlockSize =
    512;  // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;  // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152;  // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520;  // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760;  // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;  // round up large allocs to 2 MiB
constexpr size_t kUnitMB = 1024 * 1024;  // 1MiB = 1024 * 1024 bytes

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat &stat, int64_t amount) {
  stat.current += amount;
  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat &stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat &stat) { stat.peak = stat.current; }

template <typename Func>
void for_each_selected_stat_type(const StatTypes &stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(StatArray &stat_array, int64_t amount,
                       const StatTypes &stat_types) {
  for_each_selected_stat_type(stat_types,
                              [&stat_array, amount](size_t stat_type) {
                                update_stat(stat_array[stat_type], amount);
                              });
}

struct Block;
using Comparison = bool (*)(const Block *, const Block *);

static bool BlockComparatorSize(const Block *a, const Block *b);

static bool BlockComparatorAddress(const Block *a, const Block *b);

struct BlockPool {
  std::set<Block *, Comparison> blocks;
  std::set<Block *, Comparison> unmapped;
  const bool is_small;

  BlockPool(bool small)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small) {}
};

struct ExpandableSegment;

struct Block {
  int device;              // npu
  aclrtStream stream;      // allocation stream
  stream_set stream_uses;  // streams on which the block was used
  size_t size;             // block size in bytes
  size_t requested_size;   // memory originally requested
  BlockPool *pool;         // owning memory pool
  void *ptr;               // memory address
  bool allocated;          // in-use flag
  bool mapped{true};       // is the virtual address range this Block references
  // backed by physical pages. Always true when
  // expandable_segment_ is null. When false
  // This Block will be aligned to the segment size
  // of its expandable_segment_.
  Block *prev;      // prev block if split from a larger allocation
  Block *next;      // next block if split from a larger allocation
  int event_count;  // number of outstanding NPU events
  int gc_count{0};  // counter for prioritizing older / less useful blocks for
  // garbage collection
  ExpandableSegment *expandable_segment_{nullptr};

  std::shared_ptr<c10::GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when cudaMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<c10::GatheredContext> context_when_segment_allocated;

  Block(int device, aclrtStream stream, size_t size, BlockPool *pool, void *ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0),
        gc_count(0) {}

  // constructor for search key
  Block(int device, aclrtStream stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0),
        gc_count(0) {}

  bool is_split() const { return (prev != nullptr) || (next != nullptr); }

  void splice(Block *before, Block *after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

struct SegmentRange {
  char *ptr;
  size_t size;

  SegmentRange(void *p, size_t s) : ptr(static_cast<char *>(p)), size(s) {}
};

struct ExpandableSegment {
  ExpandableSegment(int device, aclrtStream stream, size_t size)
      : device_(device),
        stream_(stream),
        max_handles_(0),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size) {
    size_t device_free;
    size_t device_total;
    TORCH_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total) ==
                    ACL_ERROR_NONE,
                "aclrtGetMemInfo failed.");
    TORCH_INTERNAL_ASSERT(device_free <= device_total);
    // we allocate enough address space for 1 1/8 the total memory on the NPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    constexpr size_t extra_space_factor = 8;
    max_handles_ =
        numSegments(device_total + device_total / extra_space_factor);
    TORCH_CHECK(aclrtReserveMemAddress(&ptr_, segment_size_ * max_handles_, 0,
                                       NULL, 1) == ACL_ERROR_NONE,
                "Error, failed to reserve memory address");
  }

  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    while (end > handles_.size()) {
      handles_.emplace_back(c10::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      aclrtDrvMemHandle handle = nullptr;
      aclrtPhysicalMemProp prop = {};
      prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
      prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
      prop.memAttr = ACL_HBM_MEM_HUGE;
      prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      prop.reserve = 0;
      auto status = aclrtMallocPhysical(&handle, segment_size_, &prop, 0);
      if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = c10::nullopt;
          TORCH_CHECK(aclrtFreePhysical(h) == ACL_ERROR_NONE,
                      "aclrtFreePhysical failed.");
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
      handles_.at(i) = handle;
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_CHECK(
          aclrtMapMem((void *)((char *)ptr_ + i * segment_size_), segment_size_,
                      0, handles_.at(i).value(), 0) == ACL_ERROR_NONE,
          "Error, failed to map memory");
    }
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char *ptr() const { return (char *)ptr_; }

  size_t size() const { return max_handles_ * segment_size_; }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    TORCH_CHECK(aclrtReleaseMemAddress(ptr_) == ACL_ERROR_NONE,
                "aclrtReleaseMemAddress failed.");
  }

 private:
  void unmapHandles(size_t begin, size_t end) {
    // note: unlike aclrtFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::npu::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    TORCH_CHECK(aclrtSynchronizeStream(stream_) == ACL_ERROR_NONE,
                "aclrtSynchronizeStream failed.");
    for (auto i : c10::irange(begin, end)) {
      aclrtDrvMemHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      TORCH_CHECK(aclrtUnmapMem((void *)((char *)ptr_ + segment_size_ * i)) ==
                      ACL_ERROR_NONE,
                  "aclrtUnmapMem failed.");
      TORCH_CHECK(aclrtFreePhysical(h) == ACL_ERROR_NONE,
                  "aclrtFreePhysical failed.");
    }
    trimHandles();
  }

  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }

  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    auto start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }

  size_t segmentLeft(char *p) {
    auto size = p - ptr();
    return size / segment_size_;
  }

  size_t segmentRight(char *p) {
    auto size = p - ptr();
    return numSegments(size);
  }

  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    TORCH_INTERNAL_ASSERT(end >= begin);
    return SegmentRange(ptr() + segment_size_ * begin,
                        segment_size_ * (end - begin));
  }

  int device_;
  aclrtStream stream_;
  void *ptr_{};
  size_t max_handles_;
  size_t segment_size_;
  std::vector<c10::optional<aclrtDrvMemHandle>> handles_;
};

static bool BlockComparatorSize(const Block *a, const Block *b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
           reinterpret_cast<uintptr_t>(b->stream);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
         reinterpret_cast<uintptr_t>(b->ptr);
}

static bool BlockComparatorAddress(const Block *a, const Block *b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
           reinterpret_cast<uintptr_t>(b->stream);
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
         reinterpret_cast<uintptr_t>(b->ptr);
}

inline std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << (size / 1048576.0);
    os << " MiB";
  } else {
    os << (size / 1073741824.0);
    os << " GiB";
  }
  return os.str();
}

struct AllocParams {
  AllocParams(int device, size_t size, aclrtStream stream, BlockPool *pool,
              size_t alloc_size, DeviceStats &stats,
              std::shared_ptr<shm_alloc::memory_heap> shmem_pool = nullptr)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(ACL_ERROR_NONE),
        mem_heap_pool(shmem_pool) {}

  int device() const { return search_key.device; }

  aclrtStream stream() const { return search_key.stream; }

  size_t size() const { return search_key.size; }

  Block search_key;
  BlockPool *pool;
  size_t alloc_size;
  Block *block;
  StatTypes stat_types = {false};
  aclError err;

  std::shared_ptr<shm_alloc::memory_heap> mem_heap_pool;
};

class EventPool {
 public:
  using Event = std::unique_ptr<c10_npu::NPUEvent,
                                std::function<void(c10_npu::NPUEvent *)>>;

  // Explicit device count
  EventPool() : pools_(c10_npu::device_count()) {}

  Event get(int device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto &pool = pools_[device];
    auto destructor = [&pool](c10_npu::NPUEvent *event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<c10_npu::NPUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto *event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<c10_npu::NPUEvent>(ACL_EVENT_CAPTURE_STREAM_PROGRESS)
            .release(),
        destructor);
  }

  void empty_cache() {
    for (auto &pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<c10_npu::NPUEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

class CachingAllocatorConfig {
 public:
  static size_t max_split_size() { return instance().m_max_split_size; }

  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() { return instance().m_expandable_segments; }

  static CachingAllocatorConfig &instance() {
    static CachingAllocatorConfig *s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char *env = getenv("PYTORCH_NPU_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char *env);

 private:
  size_t m_max_split_size;
  double m_garbage_collection_threshold;
  bool m_expandable_segments;
  bool set_expandable_segments_flag = false;

  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0),
        m_expandable_segments(false) {
    // void *ptr = nullptr;
    // constexpr size_t virtual_mem_size = 512;
    // auto status = aclrtReserveMemAddress(&ptr, virtual_mem_size, 0, NULL, 1);
    // if (status == ACL_ERROR_NONE) {
    //     TORCH_CHECK(aclrtReleaseMemAddress(ptr) == ACL_ERROR_NONE,
    //     "aclrtReleaseMemAddress failed.");
    // } else {
    //     m_expandable_segments = false;
    // }
  }

  void lexArgs(const char *env, std::vector<std::string> &config);

  void consumeToken(const std::vector<std::string> &config, size_t i,
                    const char c);

  size_t parseMaxSplitSize(const std::vector<std::string> &config, size_t i);

  size_t parseGarbageCollectionThreshold(const std::vector<std::string> &config,
                                         size_t i);

  size_t parseExpandableSegments(const std::vector<std::string> &config,
                                 size_t i);
};

class DeviceCachingAllocator {
 public:
  // shm-vmm heap class for shmem
  std::shared_ptr<shm_alloc::memory_heap> mem_heap_pool_{nullptr};
  bool mem_heap_inited = false;

 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream
  ska::flat_hash_set<Block *> active_blocks;

  // allocated by shmem
  ska::flat_hash_set<void *> shmem_addrs;

  // outstanding acl events
  ska::flat_hash_map<c10_npu::NPUStream,
                     std::deque<std::pair<EventPool::Event, Block *>>>
      npu_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  // record maximum allowed memory.
  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment *> expandable_segments_;

  bool set_fraction = false;

  bool record_history = false;

  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  RecordContext record_context_ = RecordContext::NEVER;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry> *
      alloc_trace;  // pointer because we need to intentionally leak this on
                    // deallocation it can hold references to Python state which
                    // will already be destroyed when we are in exit handlers

 public:
  DeviceCachingAllocator()
      : large_blocks(false),
        small_blocks(true),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size =
        static_cast<int64_t>(CachingAllocatorConfig::max_split_size());
    context_recorder_.store(nullptr);
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<c10::GatheredContext> maybeGatherContext(
      RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block *malloc(int device, size_t orig_size, aclrtStream stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (device == -1) {
      TORCH_CHECK(c10_npu::GetDevice(&device) == ACL_ERROR_NONE,
                  "GetDevice failed.");
    }

    // process outstanding npuEvents
    process_events(context);
    auto size = round_size(orig_size);
    auto &pool = get_pool(size);

    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats,
                       mem_heap_pool_);
    params.stat_types = get_stat_types_for_pool(pool);

    bool block_found = false;
    // First, try to get a block from the existing pool.
    block_found =
        // Search pool
        get_free_block(params) ||
        // Trigger callbacks and retry search
        (trigger_free_memory_callbacks(params) && get_free_block(params));
    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(set_fraction &&
                       CachingAllocatorConfig::garbage_collection_threshold() >
                           0.0)) {
        garbage_collect_cached_blocks(context);
      }
      // Attempt allocate
      block_found = alloc_block(params, false, context, lock) ||
                    // Free enough available cached blocks to satisfy alloc
                    // and retry alloc.
                    (release_available_cached_blocks(params, context) &&
                     alloc_block(params, false, context, lock));
    }
    if (!block_found) {
      ASCEND_LOGE(
          "Get a block from the existing pool failed. Try to free "
          "cached blocks and reallocate. This error log "
          "can be ignored.");
      // Free all non-split cached blocks and retry alloc.
      block_found = (release_cached_blocks(true, context) &&
                     alloc_block(params, true, context, lock));
    }
    if (!block_found) {
      if (params.err == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        size_t device_free;
        size_t device_total;
        TORCH_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total) ==
                        ACL_ERROR_NONE,
                    "aclrtGetMemInfo failed.");
        TORCH_INTERNAL_ASSERT(device_free <= device_total);

        std::string allowed_info;
        if (set_fraction) {
          allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
        }
        stats.num_ooms += 1;
        // "total capacity": total global memory on NPU
        // "allowed": memory is allowed to use, which set by fraction.
        // "already allocated": memory allocated by the program using the
        //                      caching allocator
        // "free": free memory as reported by the NPU API
        // "cached": memory held by the allocator but not used by the program
        //
        // The "allocated" amount  does not include memory allocated outside
        // of the caching allocator, such as memory allocated by other programs
        // or memory held by the driver.
        //
        // The sum of "allocated" + "free" + "cached" may be less than the
        // total capacity due to memory held by the driver and usage by other
        // programs.
        //
        // Note that at this point free_cached_blocks has already returned all
        // possible "cached" memory to the driver. The only remaining "cached"
        // memory is split from a larger block that is partially in-use.
        AT_ERROR(
            "NPU out of memory. Tried to allocate ", format_size(alloc_size),
            " (NPU ", device, "; ", format_size(device_total),
            " total capacity; ",
            format_size(
                stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                    .current),
            " already allocated; ",
            format_size(
                stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                    .current),
            " current active; ", format_size(device_free), " free; ",
            allowed_info,
            format_size(
                stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                    .current),
            " reserved in total by PyTorch)",
            " If reserved memory is >> allocated memory try setting "
            "max_split_size_mb to avoid fragmentation.");
      } else {
        params.err;
      }
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(std::move(params), orig_size, std::move(context),
                             split_remainder);
  }

  Block *alloc_found_block(AllocParams params, size_t orig_size,
                           std::shared_ptr<c10::GatheredContext> context,
                           bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(params.err == ACL_ERROR_NONE &&
                          params.block != nullptr &&
                          params.block->ptr != nullptr);
    Block *block = params.block;
    Block *remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char *>(remaining->ptr) + size;
      remaining->size -= size;
      pool->blocks.insert(remaining);

      if (already_split && !block->expandable_segment_) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(stats.inactive_split_bytes,
                          -static_cast<std::int64_t>(block->size),
                          params.stat_types);
      } else if (!block->expandable_segment_) {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(stats.inactive_split_bytes[stat_type],
                      static_cast<std::int64_t>(remaining->size));
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }
    } else if (already_split && !block->expandable_segment_) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.inactive_split_bytes[stat_type],
                    -static_cast<std::int64_t>(block->size));
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;

    block->context_when_allocated = std::move(context);
    record_trace(TraceEntry::ALLOC, int64_t(block->ptr), orig_size,
                 block->stream, block->device, block->context_when_allocated);

    active_blocks.insert(block);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(stats.allocated_bytes[stat_type],
                  static_cast<std::int64_t>(block->size));
      update_stat(stats.active[stat_type], 1);
      update_stat(stats.active_bytes[stat_type],
                  static_cast<std::int64_t>(block->size));
      update_stat(stats.requested_bytes[stat_type],
                  static_cast<std::int64_t>(block->requested_size));
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    ASCEND_LOGD(
        "PTA CachingAllocator malloc: malloc = %zu, cached = %lu, allocated = "
        "%lu",
        block->size,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
            .current);

    return block;
  }

  void free(Block *block) {
    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*(block->pool));
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(stats.allocated_bytes[stat_type], -block->size);
    });

    record_trace(TraceEntry::FREE_REQUESTED, int64_t(block->ptr),
                 block->requested_size, block->stream, block->device,
                 context ? context : block->context_when_allocated);

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty() &&
        c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
      insert_events(block);
    } else {
      free_block(block, context);
    }

    ASCEND_LOGD(
        "PTA CachingAllocator free: free = %zu, cached = %lu, allocated = %lu",
        orig_block_size,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
            .current);
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache(bool check_error) {
    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(release_cached_blocks(check_error, context),
                "release_cached_blocks failed.");
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0;
         statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0;
         statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }

    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  std::vector<TraceEntry> trace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(result.end(), alloc_trace->begin() + alloc_trace_next,
                  alloc_trace->end());
    result.insert(result.end(), alloc_trace->begin(),
                  alloc_trace->begin() + alloc_trace_next);

    return result;
  }

  static size_t round_size(size_t size) {
    const size_t align_size = 32;
    size = size + align_size;
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block *> get_all_blocks() const {
    std::vector<const Block *> blocks;
    blocks.insert(blocks.end(), small_blocks.blocks.begin(),
                  small_blocks.blocks.end());
    blocks.insert(blocks.end(), large_blocks.blocks.begin(),
                  large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  Block *find_expandable_block(int device, aclrtStream stream, BlockPool *pool,
                               size_t size) {
    Block key(device, stream, 0);

    auto allocatable = [](Block *b) {
      return b && !b->allocated && b->event_count == 0 &&
             b->stream_uses.empty();
    };
    auto has_available_address_space = [&](Block *b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };
    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->stream == stream; ++it) {
      Block *c = *it;
      // we found the lowest address of an unmapped segment
      // but there might be a free segment we can also use
      // right before it
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        return c;
      }
    }
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    expandable_segments_.emplace_back(
        new ExpandableSegment(device, stream, segment_size));

    ExpandableSegment *es = expandable_segments_.back();
    Block *candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(Block *to_map, size_t size,
                 const std::shared_ptr<c10::GatheredContext> &ctx) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
    if (mapped_range.size == 0) {
      return false;
    }
    TORCH_INTERNAL_ASSERT(mapped_range.ptr == to_map->ptr &&
                          mapped_range.size >= size);

    BlockPool &pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      Block *remaining = new Block(
          to_map->device, to_map->stream, to_map->size - mapped_range.size,
          &pool, static_cast<char *>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment_ = to_map->expandable_segment_;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    TORCH_CHECK(try_merge_blocks(to_map, to_map->prev, pool) >= 0,
                "try_merge_blocks failed.");
    TORCH_CHECK(try_merge_blocks(to_map, to_map->next, pool) >= 0,
                "try_merge_blocks failed.");

    pool.blocks.insert(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], mapped_range.size);
    });

    record_trace(TraceEntry::SEGMENT_MAP, int64_t(mapped_range.ptr),
                 mapped_range.size, to_map->stream, to_map->device, ctx);
    if (!to_map->prev && !to_map->context_when_segment_allocated) {
      to_map->context_when_segment_allocated = ctx;
    }

    return true;
  }

  Block *try_allocate_expandable_block(
      int device, aclrtStream stream, BlockPool *pool, size_t size,
      const std::shared_ptr<c10::GatheredContext> &ctx) {
    Block *candidate = find_expandable_block(device, stream, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), ctx)) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    while (candidate->size < size) {
      // invariant: free -> unmapped -> *
      // map_block will map some of unmapped and merge with free
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(new_candidate, std::min(remaining, candidate->next->size),
                     ctx)) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
  }

  /** moves a block into a pool of cached free blocks **/
  void free_block(Block *block,
                  const std::shared_ptr<c10::GatheredContext> &context) {
    AT_ASSERT(!block->allocated && block->event_count == 0);

    record_trace(TraceEntry::FREE_COMPLETED, int64_t(block->ptr),
                 block->requested_size, block->stream, block->device,
                 context ? context : block->context_when_allocated);

    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto &pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block *, 2> merge_candidates = {block->prev, block->next};
    for (Block *merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          static_cast<int64_t>(try_merge_blocks(block, merge_candidate, pool));
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    pool.blocks.insert(block);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += static_cast<int64_t>(block->size);
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segements from
      // inactive_split
      if (!block->expandable_segment_) {
        update_stat(stats.inactive_split[stat_type],
                    net_change_inactive_split_blocks);
        update_stat(stats.inactive_split_bytes[stat_type],
                    net_change_inactive_split_size);
      }
      update_stat(stats.active[stat_type], -1);
      update_stat(stats.active_bytes[stat_type], -original_block_size);
      update_stat(stats.requested_bytes[stat_type],
                  -static_cast<std::int64_t>(requested_size));
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. **/
  size_t try_merge_blocks(Block *dst, Block *src, BlockPool &pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased =
        src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    delete src;
    src = nullptr;

    return subsumed_size;
  }

  BlockPool &get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool &pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  bool should_split(const Block *block, size_t size) {
    TORCH_INTERNAL_ASSERT(block->size >= size);
    size_t remaining = block->size - size;
    if (block->pool->is_small ||
        CachingAllocatorConfig::expandable_segments()) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
             (remaining > kSmallSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams &p) {
    BlockPool &pool = *p.pool;

    if (C10_UNLIKELY(set_fraction &&
                     CachingAllocatorConfig::garbage_collection_threshold() >
                         0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto &b : pool.blocks) {
        ++b->gc_count;
      }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      return false;
    }

    if ((*it)->expandable_segment_) {
      TORCH_WARN("enter expandable_segment_");
      if (CachingAllocatorConfig::expandable_segments()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](Block *b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->stream == p.stream() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // Rarely expandable segments has been turned off after we have
        // already allocated some blocks as expandable. For instance,
        // since we cannot share expandable memory via IPC, someone might
        // temporarily disable it. In this case we need to honor this request
        // by only finding non-expandable blocks
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
                 (*it)->stream == p.stream());
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
          return false;
        }
      }
    }

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size())) {
      return false;
    }
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer)) {
      return false;
    }
    p.block = *it;
    (*it)->gc_count = 0;  // Denote this block has been used
    pool.blocks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams &p) {
    bool freed_memory = false;
    return freed_memory;
  }

  void garbage_collect_cached_blocks(
      const std::shared_ptr<c10::GatheredContext> &ctx) {
    // Free unused cached blocks to reclaim NPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto &b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    TORCH_CHECK(c10_npu::npuSynchronizeDevice(true),
                "npuSynchronizeDevice failed.");

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block *block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count;  // Decrement the age
          freeable_block_count--;        // One less block that can be freed
          release_block(block, ctx);

          ASCEND_LOGD(
              "PTA CachingAllocator gc: free = %zu, cached = %lu, allocated = "
              "%lu",
              block->size,
              stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current,
              stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current);
        }
      }
    }
  }

  bool alloc_block(AllocParams &p, bool isRetry,
                   const std::shared_ptr<c10::GatheredContext> &ctx,
                   std::unique_lock<std::recursive_mutex> &lock) {
    size_t size = p.alloc_size;
    void *ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
    } else if (CachingAllocatorConfig::expandable_segments()) {
      TORCH_WARN("expandable_block");
      p.block = try_allocate_expandable_block(p.device(), p.stream(), p.pool,
                                              p.size(), ctx);
      if (p.block) {
        p.err = ACL_ERROR_NONE;
      } else {
        p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
      }
      return bool(p.block);
    } else {
      if (mem_heap_inited) {
        p.err = shm_alloc::aclrtMallocAlign32_mock(&ptr, size, p.mem_heap_pool);
        if (p.err == ACL_ERROR_NONE) {
          shmem_addrs.insert(ptr);
        }
      } else {
        p.err = aclrtMallocAlign32(
            &ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
      }
    }

    if (p.err != ACL_ERROR_NONE) {
      p.err = ACL_ERROR_RT_MEMORY_ALLOCATION;
      return false;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char *)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);
    ASCEND_LOGD("pta_memory acl_malloc: malloc = %zu, ret = %d", size, p.err);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    record_trace(TraceEntry::SEGMENT_ALLOC, int64_t(p.block->ptr),
                 p.block->size, p.stream(), p.device(), ctx);
    p.block->context_when_segment_allocated = ctx;
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * to satisfy the target size **/
  bool release_available_cached_blocks(
      const AllocParams &p, const std::shared_ptr<c10::GatheredContext> &ctx) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max()) {
      return false;
    }
    BlockPool &pool = *p.pool;
    Block key = p.search_key;
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
                   ? CachingAllocatorConfig::max_split_size()
                   : key.size;
    auto it = pool.blocks.lower_bound(&key);

    TORCH_CHECK(c10_npu::npuSynchronizeDevice(true),
                "npuSynchronizeDevice failed.");

    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin()) {
        return false;
      }
      size_t totalReleased = 0;
      // Back up one item.  Now on the largest block for the correct stream
      --it;
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur, ctx);
        } else {
          release_block(*cur, ctx);
          break;
        }
      }
      if (totalReleased < key.size) {
        return false;
      }
    } else {
      release_block(*it, ctx);
    }
    return true;
  }

  bool release_cached_blocks(
      bool check_error, const std::shared_ptr<c10::GatheredContext> &context) {
    // Make sure event deque from taskqueue, then synchronize Event
    TORCH_CHECK(c10_npu::npuSynchronizeDevice(check_error),
                "npuSynchronizeDevice failed.");

    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(check_error, context);

    // Free all non-split cached blocks
    release_blocks(large_blocks, context);
    release_blocks(small_blocks, context);

    return true;
  }

  void release_expandable_segment(Block *block) {
    TORCH_INTERNAL_ASSERT(block->size == block->expandable_segment_->size(),
                          "block disagrees with segment");
    TORCH_INTERNAL_ASSERT(!block->mapped);
    auto it = std::find(expandable_segments_.begin(),
                        expandable_segments_.end(), block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end());
    expandable_segments_.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment_;
    block->expandable_segment_ = nullptr;
    delete block;
    block = nullptr;
  }

  void release_block(Block *block,
                     const std::shared_ptr<c10::GatheredContext> &context) {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    record_trace(TraceEntry::SEGMENT_FREE, int64_t(block->ptr), block->size,
                 block->stream, block->device,
                 context ? context : block->context_when_segment_allocated);
    if (shmem_addrs.count((void *)block->ptr)) {
      TORCH_CHECK(shm_alloc::aclrtFree_mock((void *)block->ptr,
                                            mem_heap_pool_) == ACL_ERROR_NONE,
                  "aclrtFree failed. shmem");
    } else {
      TORCH_CHECK(aclrtFree((void *)block->ptr) == ACL_ERROR_NONE,
                  "aclrtFree failed. ori");
    }

    total_allocated_memory -= block->size;

    auto *pool = block->pool;

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(stats.reserved_bytes[stat_type], -block->size);
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);

    ASCEND_LOGD("pta_memory acl_free: free_size = %zu", block->size);

    pool->blocks.erase(block);
    delete block;
    block = nullptr;
  }

  void unmap_block(Block *block,
                   const std::shared_ptr<c10::GatheredContext> &context) {
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size =
        static_cast<char *>(unmapped.ptr) - static_cast<char *>(block->ptr);
    if (before_size > 0) {
      // prev? -> before_free -> block
      Block *before_free = new Block(block->device, block->stream, before_size,
                                     block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->blocks.insert(before_free);
    }

    TORCH_CHECK(block->size >= before_size + unmapped.size,
                "after size should be greater than or equal to 0");
    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      Block *after_free =
          new Block(block->device, block->stream, after_size, block->pool,
                    static_cast<char *>(unmapped.ptr) + unmapped.size);
      after_free->expandable_segment_ = block->expandable_segment_;
      after_free->splice(block, block->next);
      block->pool->blocks.insert(after_free);
    }

    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    TORCH_CHECK(try_merge_blocks(block, block->prev, *block->pool) >= 0,
                "try_merge_blocks failed.");
    TORCH_CHECK(try_merge_blocks(block, block->next, *block->pool) >= 0,
                "try_merge_blocks failed.");
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], -unmapped.size);
    });
    record_trace(TraceEntry::SEGMENT_UNMAP, int64_t(unmapped.ptr),
                 unmapped.size, block->stream, block->device,
                 context ? context : block->context_when_segment_allocated);
  }

  void release_blocks(BlockPool &pool,
                      const std::shared_ptr<c10::GatheredContext> &context) {
    std::vector<Block *> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block *block = *it;
      ++it;
      if (block->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block, context);
      }
    }
    for (Block *block : to_unmap) {
      unmap_block(block, context);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  EventPool::Event create_event_internal(int idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto *event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events(
      bool check_error, const std::shared_ptr<c10::GatheredContext> &context) {
    // Synchronize on outstanding events and then free associated blocks.
    for (auto &st : npu_events) {
      for (auto &e : st.second) {
        EventPool::Event event = std::move(e.first);
        Block *block = e.second;

        if (check_error) {
          TORCH_CHECK(aclrtSynchronizeEvent(*event) == ACL_ERROR_NONE,
                      "aclrtSynchronizeEvent failed.");
        } else {
          TORCH_CHECK(aclrtSynchronizeEvent(*event) == ACL_ERROR_NONE,
                      "aclrtSynchronizeEvent failed");
        }
        ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed");

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
    }

    npu_events.clear();
  }

  void insert_events(Block *block) {
    aclrtContext compiler_ctx = aclrtContext();
    aclError ret_ctx = aclrtGetCurrentContext(&compiler_ctx);

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto &stream : streams) {
      TORCH_CHECK(c10_npu::SetDevice(stream.device_index()) == ACL_ERROR_NONE,
                  "SetDevice failed.");

      EventPool::Event event = create_event_internal(stream.device_index());
      event->record(stream);
      ASCEND_LOGI("Event: record DeviceAllocator is successfully executed");

      block->event_count++;
      npu_events[stream].emplace_back(std::move(event), block);
    }
    if (ret_ctx == ACL_ERROR_NONE) {
      TORCH_CHECK(aclrtSetCurrentContext(compiler_ctx) == ACL_ERROR_NONE,
                  "aclrtSetCurrentContext failed.");
    }
  }

  void process_events(const std::shared_ptr<c10::GatheredContext> &context) {
    // Process outstanding npuEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    for (auto it = npu_events.begin(); it != npu_events.end();) {
      while (!it->second.empty()) {
        auto &e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block *block = e.second;

        if (!event->query()) {
          e.first = std::move(event);
          break;
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = npu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  void record_trace(TraceEntry::Action action, int64_t addr, size_t size,
                    aclrtStream stream, int device,
                    std::shared_ptr<c10::GatheredContext> context) {
    if (!record_history) {
      return;
    }

    auto te = TraceEntry(
        action, device, addr, size, stream,
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);

    if (record_history) {
      if (alloc_trace->size() < alloc_trace_max_entries_) {
        alloc_trace->emplace_back(te);
      } else {
        (*alloc_trace)[alloc_trace_next++] = te;
        if (alloc_trace_next == alloc_trace_max_entries_) {
          alloc_trace_next = 0;
        }
      }
    }
  }
};

void local_raw_delete(void *ptr);

void finalize();

class NpuCachingCustomAllocator {
 private:
  std::mutex mutex;

  // allocated blocks by device pointer
  ska::flat_hash_map<void *, Block *> allocated_blocks;

  void add_allocated_block(Block *block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  std::vector<void *> shm_ptr_meta;

  std::mutex *getFreeMutex() const;

  Block *get_allocated_block(void *ptr, bool remove = false);

  void setMemoryFraction(double fraction, int device);

  void init(int device_count);

  bool initialized();

  void emptyCache(bool check_error);

  DeviceStats getDeviceStats(int device);

  void resetPeakStats(int device);

  std::string name();

  void *malloc(int device, size_t size, aclrtStream stream);

  void free(void *ptr);

  void assertValidDevice(int device);
};

extern NpuCachingCustomAllocator my_allocator;

extern "C" {
EXPORT_API void *my_malloc(size_t size, int device, aclrtStream stream) {
  void *ptr = nullptr;
  if (size == 0) {
    return ptr;
  }
  ptr = my_allocator.malloc(device, size, stream);
  return ptr;
}

EXPORT_API void my_free(void *ptr, size_t size, int device,
                        aclrtStream stream) {
  my_allocator.free(ptr);
}

EXPORT_API void my_init(int device_count) { my_allocator.init(device_count); }

EXPORT_API void my_empty_cache(bool check_error) {
  my_allocator.emptyCache(true);
}

EXPORT_API DeviceStats my_get_device_stats(int device) {
  return my_allocator.getDeviceStats(device);
}

EXPORT_API void my_reset_peak_stats(int device) {
  return my_allocator.resetPeakStats(device);
}

EXPORT_API void init_shmem(int my_rank, int n_ranks, uint64_t local_mem_size,
                           uint64_t meta_size, const char *ip_port) {
  std::cout << my_rank << " " << n_ranks << " " << local_mem_size << " "
            << meta_size << " " << ip_port << std::endl;

  if (shmem_init_status() == 0) {
    auto status = shmem_set_conf_store_tls(false, nullptr, 0);
    TORCH_INTERNAL_ASSERT(status == shmem_error_code_t::SHMEM_SUCCESS,
                          "[E]shmem shmem_set_conf_store_tls error.")
    shmem_init_attr_t *attributes;
    status =
        shmem_set_attr(my_rank, n_ranks, local_mem_size, ip_port, &attributes);
    TORCH_INTERNAL_ASSERT(status == shmem_error_code_t::SHMEM_SUCCESS,
                          "[E]shmem shmem_set_attr error.")
    status = shmem_init_attr(attributes);
    TORCH_INTERNAL_ASSERT(status == shmem_error_code_t::SHMEM_SUCCESS,
                          "[E]shmem shmem_init_attr error.")
  }

  void *shmem_base_ptr = shmem_malloc(local_mem_size);
  int device = 0;
  c10_npu::GetDevice(&device);

  my_allocator.device_allocator[device]->mem_heap_inited = true;
  my_allocator.device_allocator[device]->mem_heap_pool_ =
      std::make_shared<shm_alloc::memory_heap>(shmem_base_ptr + meta_size,
                                               local_mem_size - meta_size);
}
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
// ) {
// }
