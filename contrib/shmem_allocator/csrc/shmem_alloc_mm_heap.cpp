/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "shmem_alloc_mm_heap.h"

constexpr int32_t ALIGN_32 = 32;

namespace shm_alloc {
bool range_size_first_comparator::operator()(
    const memory_range &mr1, const memory_range &mr2) const noexcept {
  if (mr1.size != mr2.size) {
    return mr1.size < mr2.size;
  }

  return mr1.offset < mr2.offset;
}

memory_heap::memory_heap(void *base, uint64_t size) noexcept
    : base_{reinterpret_cast<uint8_t *>(base)}, size_{size} {
  pthread_spin_init(&spinlock_, 0);
  address_idle_tree_[0] = size;
  size_idle_tree_.insert({0, size});
}

memory_heap::~memory_heap() noexcept { pthread_spin_destroy(&spinlock_); }

void *memory_heap::allocate(uint64_t size) noexcept {
  if (size == 0 || size > size_) {
    SHM_LOG_ERROR_MOCK("cannot allocate with size " << size);
    return nullptr;
  }

  auto aligned_size = allocated_size_align_up(size);
  memory_range anchor{0, aligned_size};

  pthread_spin_lock(&spinlock_);
  auto size_pos = size_idle_tree_.lower_bound(anchor);
  if (size_pos == size_idle_tree_.end()) {
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR_MOCK("cannot allocate with size: " << size);
    return nullptr;
  }

  auto target_offset = size_pos->offset;
  auto target_size = size_pos->size;
  auto addr_pos = address_idle_tree_.find(target_offset);
  if (addr_pos == address_idle_tree_.end()) {
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR_MOCK("offset(" << target_offset << ") size(" << target_size
                                 << ") in size tree, not in address tree.");
    return nullptr;
  }

  size_idle_tree_.erase(size_pos);
  address_idle_tree_.erase(addr_pos);
  address_used_tree_.emplace(target_offset, aligned_size);
  if (target_size > aligned_size) {
    memory_range left{target_offset + aligned_size, target_size - aligned_size};
    address_idle_tree_.emplace(left.offset, left.size);
    size_idle_tree_.emplace(left);
  }
  pthread_spin_unlock(&spinlock_);

  return base_ + target_offset;
}

void *memory_heap::aligned_allocate(uint64_t alignment,
                                    uint64_t size) noexcept {
  if (size == 0 || alignment == 0 || size > size_) {
    SHM_LOG_ERROR_MOCK("invalid input, align=" << alignment
                                               << ", size=" << size);
    return nullptr;
  }

  if ((alignment & (alignment - 1UL)) != 0) {
    SHM_LOG_ERROR_MOCK("alignment should be power of 2, but real "
                       << alignment);
    return nullptr;
  }

  uint64_t head_skip = 0;
  auto aligned_size = allocated_size_align_up(size);
  memory_range anchor{0, aligned_size};

  pthread_spin_lock(&spinlock_);
  auto size_pos = size_idle_tree_.lower_bound(anchor);
  while (size_pos != size_idle_tree_.end() &&
         !alignment_matches(*size_pos, alignment, aligned_size, head_skip)) {
    ++size_pos;
  }

  if (size_pos == size_idle_tree_.end()) {
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR_MOCK(
        "cannot allocate with size: " << size << ", alignment: " << alignment);
    return nullptr;
  }

  auto target_offset = size_pos->offset;
  auto target_size = size_pos->size;
  auto addr_pos = address_idle_tree_.find(target_offset);
  if (addr_pos == address_idle_tree_.end()) {
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR_MOCK("offset(" << target_offset << ") size(" << target_size
                                 << ") in size tree, not in address tree.");
    return nullptr;
  }
  memory_range result_range{size_pos->offset + head_skip, aligned_size};
  size_idle_tree_.erase(size_pos);
  address_idle_tree_.erase(addr_pos);

  if (head_skip > 0) {
    size_idle_tree_.emplace(memory_range{target_offset, head_skip});
    address_idle_tree_.emplace(target_offset, head_skip);
  }

  if (head_skip + aligned_size < target_size) {
    memory_range leftMR{target_offset + head_skip + aligned_size,
                        target_size - head_skip - aligned_size};
    size_idle_tree_.emplace(leftMR);
    address_idle_tree_.emplace(leftMR.offset, leftMR.size);
  }

  address_used_tree_.emplace(result_range.offset, result_range.size);
  pthread_spin_unlock(&spinlock_);

  return base_ + result_range.offset;
}

bool memory_heap::change_size(void *address, uint64_t size) noexcept {
  auto u8a = reinterpret_cast<uint8_t *>(address);
  if (u8a < base_ || u8a >= base_ + size_) {
    SHM_LOG_ERROR_MOCK("release invalid address " << address);
    return false;
  }

  if (size == 0) {
    release(address);
    return true;
  }

  auto offset = u8a - base_;
  pthread_spin_lock(&spinlock_);
  auto pos = address_used_tree_.find(offset);
  if (pos == address_used_tree_.end()) {
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR_MOCK("change size for address " << address
                                                  << " not allocated.");
    return false;
  }

  // size不变
  if (pos->second == size) {
    pthread_spin_unlock(&spinlock_);
    return true;
  }

  // 缩小size
  if (pos->second > size) {
    reduce_size_in_lock(pos, size);
    pthread_spin_unlock(&spinlock_);
    return true;
  }

  // 扩大size
  auto success = expend_size_in_lock(pos, size);
  pthread_spin_unlock(&spinlock_);

  return success;
}

int32_t memory_heap::release(void *address) noexcept {
  auto u8a = reinterpret_cast<uint8_t *>(address);
  if (u8a < base_ || u8a >= base_ + size_) {
    SHM_LOG_ERROR_MOCK("release invalid address " << address);
    return -1;
  }

  auto offset = u8a - base_;
  pthread_spin_lock(&spinlock_);
  auto pos = address_used_tree_.find(offset);
  if (pos == address_used_tree_.end()) {
    pthread_spin_unlock(&spinlock_);
    SHM_LOG_ERROR_MOCK("release address " << address << " not allocated.");
    return -1;
  }

  auto size = pos->second;
  uint64_t final_offset = static_cast<uint64_t>(offset);
  uint64_t final_size = size;
  address_used_tree_.erase(pos);

  auto prev_addr_pos = address_idle_tree_.lower_bound(offset);
  if (prev_addr_pos != address_idle_tree_.begin()) {
    --prev_addr_pos;
    if (prev_addr_pos != address_idle_tree_.end() &&
        prev_addr_pos->first + prev_addr_pos->second ==
            static_cast<uint64_t>(offset)) {
      // 合并前一个range
      final_offset = prev_addr_pos->first;
      final_size += prev_addr_pos->second;

      auto prev_addr_range = *prev_addr_pos;
      address_idle_tree_.erase(prev_addr_pos);
      size_idle_tree_.erase(
          memory_range{prev_addr_range.first, prev_addr_range.second});
    }
  }

  auto next_addr_pos = address_idle_tree_.find(offset + size);
  if (next_addr_pos != address_idle_tree_.end()) {  // 合并后一个range
    uint64_t next_addr = next_addr_pos->first;
    uint64_t next_size = next_addr_pos->second;
    final_size += next_size;
    address_idle_tree_.erase(next_addr_pos);
    size_idle_tree_.erase(memory_range{next_addr, next_size});
  }
  address_idle_tree_.emplace(final_offset, final_size);
  size_idle_tree_.emplace(memory_range{final_offset, final_size});
  pthread_spin_unlock(&spinlock_);

  return 0;
}

bool memory_heap::allocated_size(void *address, uint64_t &size) const noexcept {
  auto u8a = reinterpret_cast<uint8_t *>(address);
  if (u8a < base_ || u8a >= base_ + size_) {
    SHM_LOG_ERROR_MOCK("release invalid address " << address);
    return false;
  }

  auto offset = u8a - base_;
  bool exist = false;
  pthread_spin_lock(&spinlock_);
  auto pos = address_used_tree_.find(offset);
  if (pos != address_used_tree_.end()) {
    exist = true;
    size = pos->second;
  }
  pthread_spin_unlock(&spinlock_);

  return exist;
}

uint64_t memory_heap::allocated_size_align_up(uint64_t input_size) noexcept {
  constexpr uint64_t align_size = 16UL;
  constexpr uint64_t align_size_mask = ~(align_size - 1UL);
  return (input_size + align_size - 1UL) & align_size_mask;
}

bool memory_heap::alignment_matches(const memory_range &mr, uint64_t alignment,
                                    uint64_t size,
                                    uint64_t &head_skip) noexcept {
  if (mr.size < size) {
    return false;
  }

  if ((mr.offset & (alignment - 1UL)) == 0UL) {
    head_skip = 0;
    return true;
  }

  auto aligned_offset = ((mr.offset + alignment - 1UL) & (~(alignment - 1UL)));
  head_skip = aligned_offset - mr.offset;
  return mr.size >= size + head_skip;
}

void memory_heap::reduce_size_in_lock(
    const std::map<uint64_t, uint64_t>::iterator &pos,
    uint64_t new_size) noexcept {
  auto offset = pos->first;
  auto old_size = pos->second;
  pos->second = new_size;
  auto next_addr_pos = address_idle_tree_.find(offset + old_size);
  if (next_addr_pos == address_idle_tree_.end()) {
    address_idle_tree_.emplace(offset + new_size, old_size - new_size);
    size_idle_tree_.emplace(
        memory_range{offset + new_size, old_size - new_size});
  } else {
    auto next_size_pos = size_idle_tree_.find(
        memory_range{next_addr_pos->first, next_addr_pos->second});
    size_idle_tree_.erase(next_size_pos);
    next_addr_pos->second += (old_size - new_size);
    size_idle_tree_.emplace(
        memory_range{next_addr_pos->first, next_addr_pos->second});
  }
}

bool memory_heap::expend_size_in_lock(
    const std::map<uint64_t, uint64_t>::iterator &pos,
    uint64_t new_size) noexcept {
  auto offset = pos->first;
  auto old_size = pos->second;
  auto delta = new_size - old_size;

  auto next_addr_pos = address_idle_tree_.find(offset + old_size);
  if (next_addr_pos == address_idle_tree_.end() ||
      next_addr_pos->second < delta) {
    return false;
  }

  pos->second = new_size;
  auto next_size_pos = size_idle_tree_.find(
      memory_range{next_addr_pos->first, next_addr_pos->second});
  if (next_addr_pos->second == delta) {
    size_idle_tree_.erase(next_size_pos);
    address_idle_tree_.erase(next_addr_pos);
  } else {
    size_idle_tree_.erase(next_size_pos);
    next_addr_pos->second -= delta;
    size_idle_tree_.emplace(
        memory_range{next_addr_pos->first, next_addr_pos->second});
  }

  return true;
}

int aclrtMallocAlign32_mock(void **devPtr, size_t size,
                            std::shared_ptr<memory_heap> shmem_pool) {
  *devPtr = shmem_pool->aligned_allocate(ALIGN_32, size);
  if (*devPtr == nullptr) {
    return 1;
  } else {
    return 0;
  }
}

int aclrtFree_mock(void *devPtr, std::shared_ptr<memory_heap> shmem_pool) {
  return shmem_pool->release(devPtr);
}
}  // namespace shm_alloc
