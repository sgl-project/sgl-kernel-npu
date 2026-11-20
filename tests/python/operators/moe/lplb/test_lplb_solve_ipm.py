"""LPLB interior-point solver accuracy against the torch reference."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# The kernel factors the KKT system with an in-UB elimination sweep while the
# reference uses torch.linalg.solve (LU), so the two agree to float32
# tolerance rather than bit-for-bit. 1e-4 matches the tolerance the sglang LPLB
# NPU regression test uses for this pair.
IPM_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4)}

# Production shapes: NC = num_red_log + num_gpus and
# NV = num_red_phy + num_gpus + 2 for a 16-card EP group.
perf_shape = [(48, 82), (24, 42)]


def generate_data(shape, seed=42):
    """Generate a random standard-form LP; small shapes keep it well scaled."""
    nc, nv = shape
    torch.npu.manual_seed(seed)
    A = torch.rand(nc, nv, dtype=torch.float32)
    b = torch.rand(nc, dtype=torch.float32)
    c = torch.rand(nv, dtype=torch.float32)
    return A, b, c


@pytest.mark.parametrize(
    "shape, num_iters",
    [
        pytest.param((3, 6), 5, id="sglang-unit-test-shape"),
        pytest.param((24, 42), 5, id="one-replica-per-logical-16npu"),
        pytest.param((48, 82), 5, id="two-copies-32-logical-16npu"),
        pytest.param((17, 25), 5, id="edge-unaligned-uses-dot"),
        pytest.param((3, 6), 0, id="edge-zero-iterations-sentinel"),
        pytest.param(
            (24, 42),
            1,
            id="one-iteration-small-production",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            (48, 82),
            1,
            id="one-iteration-production",
            marks=pytest.mark.extra_checks,
        ),
        pytest.param(
            (17, 25),
            0,
            id="edge-unaligned-zero-iterations",
            marks=pytest.mark.extra_checks,
        ),
    ],
)
def test_solve_ipm(shape, num_iters):
    from sgl_kernel_npu.operators.moe import solve_ipm

    A, b, c = generate_data(shape)
    executor = check_accuracy(
        solve_ipm,
        args=(A, b, c, num_iters),
        standard=IPM_STANDARD,
    )
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and num_iters == 5),
    )


def _realistic_lplb_instance():
    """Build the A/b/c triple of a small LPLB placement problem on NPU.

    Two GPUs, six physical and four logical experts, one of them replicated on
    three physical slots. Unlike the random inputs above this instance actually
    satisfies the convergence test, so it exercises the elimination sweep
    instead of the ``0.5`` fallback.
    """
    from sgl_kernel_npu.operators.moe import lp_prep

    num_gpus, num_phy, num_logical = 2, 6, 4
    phy2log = torch.tensor([0, 1, 2, 3, 0, 0], dtype=torch.int64)
    logcnt = torch.bincount(phy2log, minlength=num_logical)
    log_single = torch.nonzero(logcnt == 1).flatten()
    log_replicated = torch.nonzero(logcnt > 1).flatten()

    log2phy = torch.full((num_logical, 3), -1, dtype=torch.int64)
    for i in range(num_logical):
        log2phy[i, 0] = i
    log2phy[0, 1] = 4
    log2phy[0, 2] = 5

    phy_single = log2phy[log_single, 0]
    phy_replicated = torch.nonzero(logcnt[phy2log] > 1).flatten()

    B_full = torch.zeros(num_gpus, num_phy, dtype=torch.float32)
    for i in range(num_gpus):
        B_full[i, i * 3 : (i + 1) * 3] = 1.0
    B1 = B_full[:, phy_single].contiguous()
    B2 = B_full[:, phy_replicated]

    C = torch.zeros((1, phy_replicated.numel()), dtype=torch.float32)
    C[0, phy2log[phy_replicated] == log_replicated[0]] = 1.0

    A_base = torch.vstack(
        [
            torch.hstack([C, torch.zeros((1, 2)), torch.zeros((1, 1))]),
            torch.hstack([B2, torch.eye(2), torch.full((2, 1), -1.0)]),
        ]
    )
    nc, nv = A_base.shape[0], A_base.shape[1] + 1
    A_full = torch.empty(nc, nv, dtype=torch.float32)
    A_full[:, : nv - 1].copy_(A_base)
    b = torch.empty(nc, dtype=torch.float32)
    t1 = torch.empty(log_single.numel(), dtype=torch.float32)

    # The last objective entry is the large penalty for the sink variable.
    c = torch.zeros(nv, dtype=torch.float32)
    c[-2] = 1.0
    c[-1] = 1000.0

    counts = torch.tensor([20.0, 5.0, 7.0, 3.0])
    lp_prep(
        A_full,
        b,
        t1,
        counts,
        log_single,
        log_replicated,
        B1,
        A_base.sum(1).contiguous(),
        impl="auto",
    )
    return A_full, b, c


def test_solve_ipm_converging_instance():
    """A converging instance must match the reference instead of the 0.5 sentinel."""
    from sgl_kernel_npu.operators.moe import solve_ipm

    A, b, c = _realistic_lplb_instance()
    expected = solve_ipm(A, b, c, 5, impl="torch_native")
    assert not bool(
        torch.all(expected == 0.5)
    ), "the reference must converge on the LPLB instance"

    got = solve_ipm(A, b, c, 5, impl="auto")
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_solve_ipm_writes_result_buffer():
    from sgl_kernel_npu.operators.moe import solve_ipm

    A, b, c = generate_data((24, 42))
    result = torch.empty(42, dtype=torch.float32)
    returned = solve_ipm(A, b, c, 5, result, impl="auto")
    assert returned.data_ptr() == result.data_ptr(), "result must be written in place"
    expected = solve_ipm(A, b, c, 5, impl="torch_native")
    torch.testing.assert_close(returned, expected, atol=1e-4, rtol=1e-4)
