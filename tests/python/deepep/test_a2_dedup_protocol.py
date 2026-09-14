"""CPU packet-protocol oracle: byte ownership, masks, ordering, and weighted expert results.
This checks the protocol independently of AscendC; it does not validate device synchronization.
"""

import collections
import json
import random
import struct

MAGIC = 0x44554431
H = 2048
FULL = H * 2 + 44
META = 44


def payload(src, t, generation):
    return struct.pack("<IIII", src, t, generation, src * 257 + t) * 256


def send(routes, world, window, generation, rng):
    writes = []
    full_remote = dedup_remote = 0
    leader_keys = set()
    assignment_keys = set()
    for src, tokens in enumerate(routes):
        active = [
            (t, k, e) for t, es in enumerate(tokens) for k, e in enumerate(es) if e >= 0
        ]
        counts = collections.Counter()
        leaders = {}
        for t, k, e in active:
            dest, local = divmod(e, 16)
            j = counts[e]
            counts[e] += 1
            leader_key = (t, dest)
            if leader_key not in leaders:
                leaders[leader_key] = (local, j)
            le, lj = leaders[leader_key]
            key = (dest, src, local, j)
            first = (le, lj) == (local, j)
            header = struct.pack("<11i", MAGIC, le, lj, 0, 0, 0, 0, 0, src, t, k)
            # Full payload belongs only to the leader; nonleaders overwrite metadata only.
            writes.append((key, header, payload(src, t, generation) if first else None))
            assignment_keys.add(key)
            if first:
                leader_keys.add((dest, src, t))
            if dest != src:
                full_remote += FULL
                dedup_remote += FULL if first else META
    rng.shuffle(writes)  # All writes complete before the modeled sender-ready barrier.
    for key, header, data in writes:
        old = window.get(key, (b"", b"poison"))[1]
        window[key] = (header, data if data is not None else old)
    return assignment_keys, full_remote, dedup_remote, len(leader_keys)


def receive(routes, world, window, generation):
    counts = collections.Counter()
    expected = collections.defaultdict(list)
    for src, tokens in enumerate(routes):
        for t, es in enumerate(tokens):
            for k, e in enumerate(es):
                if e >= 0:
                    dst, local = divmod(e, 16)
                    expected[dst, src, local].append(
                        (payload(src, t, generation), (src, t, k))
                    )
    outputs = collections.defaultdict(float)
    reference = collections.defaultdict(float)
    for dst in range(world):
        for local in range(16):
            for src in range(world):
                for j, (expected_x, expected_triple) in enumerate(
                    expected[dst, src, local]
                ):
                    raw, _ = window[dst, src, local, j]
                    magic, le, lj, _, _, _, _, _, s, t, k = struct.unpack("<11i", raw)
                    assert magic == MAGIC and s == src
                    got = window[dst, src, le, lj][1]
                    assert got == expected_x and (s, t, k) == expected_triple
                    counts[dst] += 1
                    # Expert-dependent result and nonuniform gate weights detect wrong expert/top-k association.
                    e = dst * 16 + local
                    v = (src + 1) * 19 + (t + 1) * 3
                    f = (e + 1) * v + (e % 7) ** 2
                    w = (k + 1) / 36
                    outputs[src, t] += w * f
    for src, tokens in enumerate(routes):
        for t, es in enumerate(tokens):
            for k, e in enumerate(es):
                if e >= 0:
                    v = (src + 1) * 19 + (t + 1) * 3
                    reference[src, t] += (k + 1) / 36 * ((e + 1) * v + (e % 7) ** 2)
    assert outputs.keys() == reference.keys()
    assert all(abs(outputs[x] - reference[x]) < 1e-7 for x in reference)
    return sum(counts.values())


def main():
    rng = random.Random(20260914)
    cases = assignments = 0
    rows = []
    for world in [2, 4, 8]:
        for T in [1, 6, 32, 128]:
            window = {}
            for gen in range(12):
                routes = []
                for src in range(world):
                    tokens = []
                    # Include heterogeneous token counts and empty sources.
                    nt = T if gen < 8 else rng.randrange(T + 1)
                    for t in range(nt):
                        if gen == 0:
                            es = [((src + 1) % world) * 16 + k for k in range(8)]
                        elif gen == 1:
                            es = [((src + k) % world) * 16 + k for k in range(8)]
                        elif gen == 2:
                            es = [src * 16 + k for k in range(8)]
                        else:
                            es = rng.sample(range(world * 16), 8)
                        if gen == 3:
                            es = [-1] * 8
                        elif gen >= 4:
                            es = [e if rng.random() > 0.35 else -1 for e in es]
                        tokens.append(es)
                    routes.append(tokens)
                keys, old, new, leaders = send(routes, world, window, gen, rng)
                n = receive(routes, world, window, gen)
                assert n == len(keys)
                assert new <= old
                if gen == 0:
                    assert leaders == world * T and new == world * T * (FULL + 7 * META)
                cases += 1
                assignments += n
                rows.append(
                    {
                        "world": world,
                        "max_tokens": T,
                        "case": gen,
                        "assignments": n,
                        "full_payloads_all_destinations": leaders,
                        "native_requested_remote_copy_bytes": old,
                        "dedup_requested_remote_copy_bytes": new,
                    }
                )
    result = {
        "valid": True,
        "cases": cases,
        "validated_assignments": assignments,
        "masked_cases": True,
        "reused_windows": True,
        "shuffled_write_completion_order": True,
        "heterogeneous_source_counts": True,
        "weighted_expert_specific_combine": True,
        "limits": "CPU protocol model; not AscendC or physical DMA validation",
        "cases_detail": rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "cases_detail"}))


def test_packet_protocol():
    main()


if __name__ == "__main__":
    main()
