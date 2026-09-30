# Fused Timestamp LRU Metadata Update

The same kernel implementation also exposes
`fused_timestamp_lru_metadata_update_with_probation`. It adds a scalar
`probation_age` input. Hits still become MRU at age zero, while newly filled
miss slots start at `probation_age` and are stably inserted into the existing
descending-age order. `probation_age=0` is behaviorally equivalent to the
original operator.

## 1. Scope

This AIV-only operator is specialized for the sparse KV configuration used by
SGLang NPU DSA:

- `topk = 2048`
- `cache_capacity ∈ {2048, 4096, 6144, 8192}`
- all metadata uses contiguous `int32`
- valid request IDs start at row `0`
- one AIV owns one request row during victim selection (grid-stride when
  `B > blockDim`); metadata writes are tiled across AIVs
- A2/A3 scatter is not used

The selection operator returns `victim_slots[B, 2048]` and
`miss_counts[B]`, and updates these tensors in place:

- `device_lru_slots[R, cache_capacity]`
- `device_lru_slot_stamps[R, cache_capacity]`

The following `parallel_lru_metadata_write` operator consumes both outputs and
updates these tensors in place:

- `slot_map[R_map, W]`
- `device_slot_tokens[R, cache_capacity]`

`device_lru_slots[row, i]` and `device_lru_slot_stamps[row, i]` are an aligned
pair. The persistent pair array is ordered by descending stamp: oldest first,
most recently hit/filled last.

## 2. Per-request algorithm

1. Load `lru_slots`, `lru_stamps`, and the capacity-sized physical-slot hit mask
   produced by `slot_map_lookup` into UB.
2. Saturating SIMD increment:
   `stamp = min(stamp, stamp_max - 1) + 1`.
3. Gather the hit bit by physical slot into the existing LRU order.
4. Use `CompareScalar` to generate packed non-hit and hit masks, then use
   `GatherMask<int32_t>` to stably compact the two groups into adjacent UB
   regions. One indexed `Gather` materializes `[non-hit, hit]`; non-hit stamps
   are compacted with `GatherMask` and their tail is zeroed by a vector prefix
   predicate. Since the persistent input is already in descending timestamp
   order and the age increment is monotonic, preserving order within both
   groups produces the same order as the former stable sort.
5. Reuse every capacity-sized buffer above the compacted LRU pair as the
   top-k victim-selection scratch arena.
6. Build the valid-miss vector with clamped SIMD arithmetic. Run an 11-round
   Hillis-Steele inclusive scan; each shift is an indexed `Gather`.
7. Gather `lru_slots[miss_rank]` and restore `-1` for non-miss positions.
8. Write one `miss_count` value per valid request for the following metadata
   kernel.
9. Binary-search the already sorted survivor suffix for the stable
   `probation_age` insertion point. Build the insertion permutation with vector
   arithmetic and gather the complete slot/stamp output pair. For the original
   operator, `probation_age=0`, so the insertion point is known directly and
   no binary search is required. Write the full rows back to GM and wait for
   MTE3 completion before the core reuses UB for another request.
10. Return `victim_slots` and `miss_counts` to the caller. The caller launches
    `parallel_lru_metadata_write` on the same stream. It splits every
    request into 64 independent 32-position tiles and distributes the tiles
    across all available AIVs. A tile copies its `topk_indices` and
    `victim_slots` into UB, loads aligned 32-byte reverse-map lines for its
    victims, and gathers all old tokens with SIMD. Valid misses are compacted
    into a double-buffered output queue, and its MTE3 stage issues these sparse
    writes:
    - `slot_map[old_token] = -1` when the victim was occupied;
    - `device_slot_tokens[victim] = new_token`;
    - `slot_map[new_token] = victim`.
    Requests with `miss_count == 0` skip their tiles before the UB copies. A
    two-entry input queue prefetches top-k/victim tiles, while a two-entry
    output queue keeps sparse-write source data alive until MTE3 completes.
    Once one output tile is launched, its writes overlap the next tile's
    reverse-map loads and Gather; there is no per-eight-write MTE3 wait. Tiles
    whose `victim_slots` are all invalid skip reverse-map loads and Gather.

Duplicate hits are safe because `slot_map_lookup` writes a binary mask with
atomic max. Top-k token IDs are expected to be unique for misses; this is
already guaranteed by the upstream top-k selector.

## 3. UB plan

The host reads the platform UB size, reserves 8 KiB for pipe overhead, and
allocates an exact capacity-dependent work arena for either LRU variant:

| Region | Bytes | Lifetime |
|---|---:|---|
| LRU slots + stamps | 32,768 | full request |
| physical hit mask | 16,384 | input; then compacted non-hit slots |
| gather offsets | 16,384 | hit gather; then compacted hit slots |
| gathered hit flags | 16,384 | masks, predicates, and final slot offsets |
| packed hit/non-hit masks | 1,024 | `CompareScalar` → `GatherMask` |
| seven top-k vectors | 57,344 | victim plan; overlays the dead mask/offset/flag area |

The stage sizes are `20.25 * cache_capacity` bytes for stable compaction,
`8 * cache_capacity + 57,344` bytes for victim selection, and
`20 * cache_capacity` bytes for writeback. The host allocates the maximum:

| Cache capacity | Work arena | With 8 KiB reserve |
|---:|---:|---:|
| 2,048 | 73,728 | 81,920 |
| 4,096 | 90,112 | 98,304 |
| 6,144 | 124,416 | 132,608 |
| 8,192 | 165,888 | 174,080 |

The 8,192-capacity plan fits a 192 KiB UB. The parallel metadata kernel uses
about 6.1 KiB of UB per AIV: 512 bytes for two prefetched input tiles,
4,352 bytes for two sparse-write records, 1,312 bytes for reverse-map
lines/Gather metadata, and 32 bytes for the constant `-1` DMA source.

## 4. Stream contract

`slot_map_lookup` runs on the caller stream. A `copy_ready` event releases one
copy stream, which runs the following kernels serially:

- D2D hit copy with 48 AIVs;
- H2D host-miss copy with 48 AIVs.

After both copies complete, the caller launches victim selection followed by
the parallel metadata-write kernel on its own stream while it prepares sparse
attention. Same-stream ordering makes
`victim_slots` and `miss_count` visible to the second kernel without a host
synchronization. Refill waits for metadata completion, then uses
`victim_slots` as destination indices. Invalid request rows in `victim_slots`
are left undefined and are ignored by the refill valid mask.

## 5. Invariants and validation

- Request IDs in one launch are unique; valid IDs lie in `[0, R)`.
- `lru_slots` is a permutation of `[0, cache_capacity)`.
- stamps lie in `[0, stamp_max]` and are non-increasing after writeback.
- `probation_age` lies in `[0, stamp_max]`.
- `slot_map[token] == slot` iff `device_slot_tokens[slot] == token` for occupied
  slots.
- Victim slots and valid miss tokens are unique within one request, so metadata
  tiles write disjoint reverse-map and slot-map entries without atomics.
- hit and invalid top-k positions of valid requests contain `-1`; output rows
  for invalid request IDs are undefined.
- `stamp_max` fits positive int32. The kernel performs no floating-point key
  conversion and no full-record sort.
