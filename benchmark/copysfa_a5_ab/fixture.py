"""Deterministic fixtures shared by native A5 and LIM/CopySFA arms.

The pure-Python helpers are intentionally importable without Torch so the CLI and
result contract can be checked on a CPU-only workstation. ``build_torch_fixture``
is called only inside an A5 arm process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

PAGE = 128
TOPK = 2048
ROW_BYTES = 656
COPY_CAPACITY = 32768


def parse_miss_counts(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("miss counts must be comma-separated integers") from error
    if not result or any(count < 0 for count in result):
        raise ValueError("miss counts must contain non-negative integers")
    return result


@dataclasses.dataclass(frozen=True)
class FixtureSpec:
    query_width: int
    batch: int
    heads: int
    index_heads: int
    capacity: int
    prefix: int
    context: int
    topk: int = TOPK
    page_size: int = PAGE
    miss_counts: tuple[int, ...] = (512, 513)
    seed: int = 20261116
    selection: str = "shared"

    def validate(self) -> None:
        if self.query_width not in (1, 4):
            raise ValueError("query_width must be 1 or 4")
        if self.selection not in ("shared", "disjoint"):
            raise ValueError("selection must be shared or disjoint")
        if self.batch < 1 or self.heads < 1 or self.index_heads < 1:
            raise ValueError("batch, heads and index_heads must be positive")
        if self.topk != TOPK or self.page_size != PAGE:
            raise ValueError("this A5 harness fixes TopK=2048 and page_size=128")
        if (
            self.capacity % PAGE
            or not self.query_width * TOPK <= self.capacity <= 32384
        ):
            raise ValueError("capacity must be 128-aligned and in [Q*2048, 32384]")
        if self.prefix % PAGE or self.prefix <= self.capacity:
            raise ValueError("prefix must be page-aligned and greater than capacity")
        if not self.prefix < self.context <= self.prefix + 256:
            raise ValueError("context must be in (prefix, prefix + 256]")
        if self.context - self.query_width + 1 < TOPK:
            raise ValueError("every route must expose at least TopK tokens")
        if not self.miss_counts or any(value < 0 for value in self.miss_counts):
            raise ValueError("miss_counts must contain non-negative integers")
        if self.prefix - max(self.miss_counts) < self.capacity:
            raise ValueError(
                "prefix cannot fill C resident rows while keeping the requested misses absent"
            )
        prefix_union = set().union(*self.route_prefix_id_sets())
        if max(self.miss_counts) > len(prefix_union):
            raise ValueError("requested misses exceed the selected prefix union")

    def route_ids(self, route: int) -> tuple[int, ...]:
        if not 0 <= route < self.query_width:
            raise ValueError("route is outside query_width")
        if self.selection == "disjoint":
            stride = 17
            while math.gcd(stride, self.prefix) != 1:
                stride += 2
            return tuple(
                (self.prefix // 3 + (route * self.topk + rank) * stride) % self.prefix
                for rank in range(self.topk)
            )
        visible = self.context - (self.query_width - 1 - route)
        return tuple(range(visible - self.topk, visible))

    def route_prefix_id_sets(self) -> list[set[int]]:
        return [
            {source for source in self.route_ids(route) if source < self.prefix}
            for route in range(self.query_width)
        ]

    def request_miss_count(self, request: int) -> int:
        return self.miss_counts[request % len(self.miss_counts)]

    def manifest(self) -> dict[str, Any]:
        self.validate()
        routes = [self.route_ids(route) for route in range(self.query_width)]
        prefix_union = sorted(
            set().union(
                *({source for source in ids if source < self.prefix} for ids in routes)
            )
        )
        return {
            "schema_version": 1,
            **dataclasses.asdict(self),
            "miss_counts": [self.request_miss_count(row) for row in range(self.batch)],
            "request_slots": list(range(1, self.batch + 1)),
            "positions": [
                self.context - self.query_width + route
                for route in range(self.query_width)
            ],
            "selected_prefix_union": len(prefix_union),
            "selected_prefix_hit_rates": [
                1 - self.request_miss_count(row) / len(prefix_union)
                for row in range(self.batch)
            ],
            "route_prefix_counts": [
                sum(source < self.prefix for source in ids) for ids in routes
            ],
            "tail_tokens": self.context - self.prefix,
            "packed_c8_row_bytes": ROW_BYTES,
            "index_score_pattern": (
                "route_partitioned_positive_v1"
                if self.selection == "disjoint"
                else "bf16_boundary_gaps_v1"
            ),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _packed_rows(torch: Any, count: int) -> Any:
    """Construct valid deterministic [FP8 C512 | BF16 KPE64 | FP32 scales4]."""
    positions = torch.arange(count, dtype=torch.int64)[:, None]
    nope = (
        ((positions * 17 + torch.arange(512)[None] * 7) % 15 - 7).float() / 4 + 0.5
    ).to(torch.float8_e4m3fn)
    rope = ((positions * 13 + torch.arange(64)[None] * 3) % 17 - 8).float() / 32
    for digit in range(3):
        rope[:, digit] = (((positions[:, 0] >> (8 * digit)) & 255).float() - 128) / 128
    scales = ((positions % 7 + torch.arange(4)[None] + 1).float() / 8).contiguous()
    packed = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.to(torch.bfloat16).contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed.shape[-1] != ROW_BYTES:
        raise AssertionError("packed C8 row must contain 656 bytes")
    return packed.view(torch.int8).contiguous()


def unpack_rows(torch: Any, rows: Any) -> tuple[Any, Any]:
    raw = rows.contiguous().view(torch.uint8)
    nope = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    rope = raw[:, 512:640].contiguous().view(torch.bfloat16).float()
    scales = raw[:, 640:].contiguous().view(torch.float32).repeat_interleave(128, -1)
    value = (nope * scales).to(torch.bfloat16).float()
    return torch.cat((value, rope), dim=-1), value


def _tensor_fingerprint(torch: Any, tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(cpu.shape), str(cpu.dtype))).encode("utf-8"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _index_key_scales(torch: Any, spec: FixtureSpec, source_capacity: int) -> Any:
    # Keep every route's TopK boundary distinct even after BF16 score sorting.
    # A linear FP32 ramp alone collapses adjacent scores at long contexts.
    scales = torch.arange(1, source_capacity + 1, dtype=torch.float32)
    if spec.selection == "disjoint":
        return (1 + scales / source_capacity).view(source_capacity // PAGE, PAGE, 1)
    scales.div_(source_capacity * 8)
    first = spec.context - (spec.query_width - 1) - TOPK
    last = spec.context - TOPK
    for route, token in enumerate(range(first, last + 1)):
        scales[token] = 0.25 * (2**route)
    scales[last + 1 :] = torch.linspace(
        4, 8, source_capacity - last - 1, dtype=torch.float32
    )
    return scales.view(source_capacity // PAGE, PAGE, 1)


def build_torch_fixture(torch: Any, spec: FixtureSpec) -> dict[str, Any]:
    """Materialize the shared fixture on CPU; callers move tensors to A5."""
    spec.validate()
    source_capacity = math.ceil(spec.context / PAGE) * PAGE
    source_pages = source_capacity // PAGE
    hbm_tokens = spec.capacity + 256
    hbm_pages = hbm_tokens // PAGE
    packed = _packed_rows(torch, source_capacity).view(source_pages, PAGE, 1, ROW_BYTES)

    routes = [list(spec.route_ids(route)) for route in range(spec.query_width)]
    prefix_union = sorted(
        set().union(
            *({source for source in ids if source < spec.prefix} for ids in routes)
        )
    )
    query_rows = spec.batch * spec.query_width
    query_ends = torch.arange(
        spec.query_width,
        query_rows + 1,
        spec.query_width,
        dtype=torch.int32,
    )
    seq_lens = torch.full((spec.batch,), spec.context, dtype=torch.int32)
    positions = torch.tensor(
        [
            spec.context - spec.query_width + route
            for _ in range(spec.batch)
            for route in range(spec.query_width)
        ],
        dtype=torch.int64,
    )

    row = torch.arange(query_rows, dtype=torch.float32)[:, None, None]
    head = torch.arange(spec.heads, dtype=torch.float32)[None, :, None]
    feature = torch.arange(576, dtype=torch.float32)[None, None, :]
    merged_query = (((row * 11 + head * 7 + feature * 3) % 31) - 15).to(
        torch.bfloat16
    ) / 16
    q = merged_query[..., :512].contiguous()
    q_rope = merged_query[..., 512:].contiguous()

    index_query = torch.ones(query_rows, spec.index_heads, 128, dtype=torch.float32)
    index_key = torch.ones(source_pages, PAGE, 1, 128, dtype=torch.float32)
    if spec.selection == "disjoint":
        index_query.zero_()
        index_key.zero_()
        group = 128 // spec.query_width
        flat_key = index_key.view(-1, 128)
        for route in range(spec.query_width):
            start, end = route * group, (route + 1) * group
            index_query[route :: spec.query_width, :, start:end] = 1
            flat_key[torch.tensor(routes[route]), start:end] = 1
    index_query = index_query.to(torch.float8_e4m3fn)
    index_key = index_key.to(torch.float8_e4m3fn)
    index_weights = torch.full(
        (query_rows, spec.index_heads), 1 / spec.index_heads, dtype=torch.bfloat16
    )
    query_scale = torch.ones(query_rows, spec.index_heads, dtype=torch.float32)
    key_scale = _index_key_scales(torch, spec, source_capacity)
    source_table = torch.arange(source_pages, dtype=torch.int32).repeat(spec.batch, 1)

    source_ids = torch.full((query_rows, 1, TOPK), -1, dtype=torch.int32)
    destination_slots = torch.full_like(source_ids, -1)
    route_misses = torch.zeros(query_rows, dtype=torch.int32)
    miss_sources = torch.full((spec.batch, 1, COPY_CAPACITY), -1, dtype=torch.int32)
    miss_slots = torch.full_like(miss_sources, -1)
    miss_counts = torch.empty(spec.batch, dtype=torch.int32)
    slot_pool_initial = torch.full(
        (spec.batch + 1, source_capacity), -1, dtype=torch.int32
    )
    initial_hbm = torch.full(
        (spec.batch * hbm_pages, PAGE, 1, ROW_BYTES), -91, dtype=torch.int8
    )
    expected_hbm = initial_hbm.clone()
    hbm_table = torch.arange(spec.batch * hbm_pages, dtype=torch.int32).view(
        spec.batch, hbm_pages
    )
    hbm_lengths = torch.full(
        (spec.batch,), spec.capacity + spec.context - spec.prefix, dtype=torch.int32
    )
    # Match SGLang's request-pool convention: row 0 is reserved for graph
    # padding and live requests occupy rows 1..B.
    request_slots = torch.arange(1, spec.batch + 1, dtype=torch.int32)

    packed_flat = packed.view(-1, ROW_BYTES)
    initial_flat = initial_hbm.view(-1, ROW_BYTES)
    expected_flat = expected_hbm.view(-1, ROW_BYTES)
    golden = torch.empty(query_rows, spec.heads, 512, dtype=torch.float32)

    prefix_union_set = set(prefix_union)
    miss_order = prefix_union
    if spec.selection == "disjoint":
        # Spread request-union misses evenly across routes, including full 8192.
        miss_order = [source for group in zip(*routes) for source in group]
    filler_candidates = [
        source for source in range(spec.prefix) if source not in prefix_union_set
    ]
    for request in range(spec.batch):
        miss_count = spec.request_miss_count(request)
        misses = miss_order[:miss_count]
        miss_set = set(misses)
        selected_hits = [source for source in prefix_union if source not in miss_set]
        filler = filler_candidates[: spec.capacity - len(selected_hits)]
        residents = selected_hits + filler
        if len(residents) != spec.capacity:
            raise AssertionError("fixture could not fill the C-pool")
        victim_slots = list(range(spec.capacity - miss_count, spec.capacity))
        post_map = {source: slot for slot, source in enumerate(residents)}
        for source, slot in zip(misses, victim_slots):
            post_map[source] = slot
        slot_pool_initial[request + 1, residents] = torch.arange(
            spec.capacity, dtype=torch.int32
        )
        miss_sources[request, 0, :miss_count] = torch.tensor(misses, dtype=torch.int32)
        miss_slots[request, 0, :miss_count] = torch.tensor(
            victim_slots, dtype=torch.int32
        )
        miss_counts[request] = miss_count

        for slot, source in enumerate(residents):
            physical = int(hbm_table[request, slot // PAGE]) * PAGE + slot % PAGE
            initial_flat[physical].copy_(packed_flat[source])
            expected_flat[physical].copy_(packed_flat[source])
        for source, slot in zip(misses, victim_slots):
            physical = int(hbm_table[request, slot // PAGE]) * PAGE + slot % PAGE
            expected_flat[physical].copy_(packed_flat[source])
        for source in range(spec.prefix, spec.context):
            slot = spec.capacity + source - spec.prefix
            physical = int(hbm_table[request, slot // PAGE]) * PAGE + slot % PAGE
            initial_flat[physical].copy_(packed_flat[source])
            expected_flat[physical].copy_(packed_flat[source])

        for route, canonical in enumerate(routes):
            prefix_ids = [source for source in canonical if source < spec.prefix]
            tail_ids = [source for source in canonical if source >= spec.prefix]
            route_miss_ids = [source for source in prefix_ids if source in miss_set]
            ordered = (
                route_miss_ids
                + [source for source in prefix_ids if source not in miss_set]
                + tail_ids
            )
            slots = [
                (
                    post_map[source]
                    if source < spec.prefix
                    else spec.capacity + source - spec.prefix
                )
                for source in ordered
            ]
            index = request * spec.query_width + route
            source_ids[index, 0] = torch.tensor(ordered, dtype=torch.int32)
            destination_slots[index, 0] = torch.tensor(slots, dtype=torch.int32)
            route_misses[index] = len(route_miss_ids)
            key, value = unpack_rows(torch, packed_flat[torch.tensor(ordered)])
            scores = merged_query[index].float() @ key.T / math.sqrt(576)
            golden[index] = torch.softmax(scores, dim=-1) @ value

    canonical_ids = torch.tensor(
        [routes[route] for _ in range(spec.batch) for route in range(spec.query_width)],
        dtype=torch.int32,
    ).view(query_rows, 1, TOPK)
    tensors = {
        "q": q,
        "q_rope": q_rope,
        "merged_query": merged_query,
        "index_query": index_query,
        "index_key": index_key,
        "index_weights": index_weights,
        "query_scale": query_scale,
        "key_scale": key_scale,
        "source_table": source_table,
        "query_ends": query_ends,
        "seq_lens": seq_lens,
        "positions": positions,
        "canonical_ids": canonical_ids,
        "source_ids": source_ids,
        "destination_slots": destination_slots,
        "route_misses": route_misses,
        "miss_sources": miss_sources,
        "miss_slots": miss_slots,
        "miss_counts": miss_counts,
        "slot_pool_initial": slot_pool_initial,
        "initial_hbm": initial_hbm,
        "expected_hbm": expected_hbm,
        "hbm_table": hbm_table,
        "hbm_lengths": hbm_lengths,
        "request_slots": request_slots,
        "packed": packed,
        "golden": golden,
    }
    return {
        "spec": spec,
        "manifest": spec.manifest(),
        "spec_sha256": spec.fingerprint(),
        "tensor_sha256": _tensor_fingerprint(torch, tensors),
        "tensors": tensors,
    }
