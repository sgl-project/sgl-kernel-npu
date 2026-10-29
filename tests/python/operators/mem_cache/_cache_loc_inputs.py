"""Representative nonconflicting inputs for cache-location scatter and gather."""

import pytest
import torch

CACHE_LOC_CASES = [
    pytest.param((8, 64), (1,), 0, False, id="minimum"),
    pytest.param((4, 32), (2, 1, 3), 2, False, id="mixed-with-tail"),
    pytest.param(
        (64, 256), (0, 1, 0, 16, 3) * 7 + (2, 1), 4, False, id="batch-tail-37"
    ),
    # SGLang speculative cache_locs uses packed B * draft_token_num int32 slots.
    pytest.param((512, 8192), (1,) * 64, 0, False, id="speculative-step-1"),
    pytest.param((512, 8192), (16,) * 128, 0, False, id="maximum-step"),
    # Exercise a batch of 300 requests with alternating interval lengths.
    pytest.param((2000, 16384), (1, 2) * 150, 0, False, id="batch-300-mixed"),
    pytest.param((32, 128), (4,) * 16, 0, True, id="matrix-locations"),
    # None selects B300 random step1/2 data across the full start range.
    pytest.param((2000, 16384), None, 0, False, id="batch-300-width-16384-random"),
    pytest.param((2000, 8192), None, 0, False, id="batch-300-width-8192-random"),
]

# Batch and interval-length combinations for changed-value round trips.
ROUNDTRIP_CASES = [
    (1, 4),
    (9, 4),
    (15, 4),
    (16, 4),
    (64, 1),
    (300, 2),
    (37, 3),
    (128, 16),
]


def generate_test_data(pool_shape, lengths, tail, matrix, req_dtype):
    """Generate random intervals; bounded cases keep 16-slot row reads in range."""
    torch.npu.manual_seed(42)
    device = torch.device("npu", torch.npu.current_device())
    rows, width = pool_shape
    if lengths is None:
        # B300 random step1/2 workload, including starts near the row end.
        requests = torch.arange(300, device=device, dtype=req_dtype)
        pool = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
        starts = torch.randint(2, width, (300,), device=device, dtype=torch.int64) - 2
        ends = starts + torch.randint(1, 3, (300,), device=device, dtype=torch.int64)
        locations = torch.randint(
            0,
            10000,
            (int((ends - starts).sum().item()),),
            device=device,
            dtype=torch.int32,
        )
        return requests, pool, starts, ends, locations
    requests = torch.randperm(rows, device=device)[: len(lengths)].to(req_dtype)
    pool = torch.arange(rows * width, device=device, dtype=torch.int32).view(pool_shape)
    starts = torch.randint(
        0, width - 16 + 1, (len(lengths),), device=device, dtype=torch.int64
    )
    ends = starts + torch.tensor(lengths, device=device, dtype=torch.int64)
    count = sum(lengths)
    locations = torch.full((count + tail,), -777, device=device, dtype=torch.int32)
    # Different source/destination values make a missing write observable.
    locations[:count] = -1000000 - torch.arange(count, device=device, dtype=torch.int32)
    if matrix:
        locations = locations.view(8, 8)
    return requests, pool, starts, ends, locations


def generate_roundtrip_data(batch, step, req_dtype, seed=0, *, device="npu"):
    """Seeded random inputs, independent of other test suites."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, width = 512, 8192
    pool = torch.randint(
        0, 1 << 20, (rows, width), generator=generator, dtype=torch.int32, device="cpu"
    ).to(device)
    requests = torch.randperm(rows, generator=generator, device="cpu")[:batch].to(
        device=device, dtype=req_dtype
    )
    starts = torch.randint(
        0, width - step, (batch,), generator=generator, dtype=torch.int64, device="cpu"
    ).to(device)
    ends = starts + step
    locations = torch.empty((batch * step,), dtype=pool.dtype, device=device)
    return pool, requests, starts, ends, locations
