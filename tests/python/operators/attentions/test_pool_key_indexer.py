"""Direct ACLNN binding tests; requires the attentions wheel and vendor op.

Run explicitly after building attentions. The kernels-only CI cannot load this
extension and deliberately does not include this file in its attention group.
"""

import json
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check
from tests.python.utils.performance import benchmark_npu, write_result

# W: synthetic one-token-per-request PA workload, using GLM-5.3-Flash's
# 32 index heads, 128 channels and pool=4; not a captured production tensor.
perf_shape = [(16, 512, 32)]
_REQUEST = ContextVar("pool_key_indexer_test_request")


@pytest.fixture(autouse=True)
def binding_options(request):
    token = _REQUEST.set(request)
    try:
        yield
    finally:
        _REQUEST.reset(token)


def _check_binding_performance(executor, enable_perf):
    """Native bindings use lower-level profiling, without Python registration."""
    request = _REQUEST.get()
    override = request.config.getoption("enable_perf")
    if not (enable_perf if override is None else override):
        return
    config = request.config
    parameters = request.node.callspec.params
    directory = (
        Path(config.getoption("prof_dir") or "profilings")
        / "prof_pool_key_indexer"
        / uuid4().hex
    )
    average_us = benchmark_npu(executor, directory)
    result = dict(
        test_case=request.node.nodeid,
        operator="attentions.pool_key_indexer",
        impl=executor.impl,
        impl_ref=executor.impl_ref,
        accuracy_checked=executor.accuracy_checked,
        parameters=json.dumps(parameters, default=str, sort_keys=True),
        profiling_dir=str(directory.resolve()),
        average_device_task_us=average_us,
        device=str(torch.device("npu", torch.npu.current_device())),
        device_name=torch.npu.get_device_name(),
        warmup=5,
        active=5,
    )
    csv_path = config.getoption("csv_path")
    if csv_path is not None:
        write_result(csv_path, **result)
    record = request.getfixturevalue("record_property")
    for key, value in result.items():
        record(key, value)
    print(
        f"pool_key_indexer [aclnn]: {average_us:.3f} us, accuracy_checked={executor.accuracy_checked}"
    )


def generate_test_data(layout, rows, n_pools, heads, dtype, topk=16):
    """Positive, separated scores make pool ranking independent of roundoff."""
    pool_size = 4
    q = torch.zeros(rows, 1, heads, 128, dtype=dtype)
    q[..., 0] = 1
    weights = torch.ones(rows, 1, heads, dtype=dtype)
    keys = torch.zeros(rows, n_pools, 1, 128, dtype=dtype)
    keys[..., 0] = torch.arange(1, n_pools + 1, dtype=dtype).view(1, n_pools, 1)
    inputs = dict(
        query=q,
        pool_key=keys,
        weights=weights,
        pool_tail_k=[3] * rows,
        topk=topk,
        pool_size=pool_size,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        quant_mode=-1,
        return_value=False,
    )
    if layout == "TND":
        inputs.update(
            query=q.squeeze(1),
            weights=weights.squeeze(1),
            layout_q="TND",
            layout_k="PA_BBND",
            actual_seq_q=list(range(1, rows + 1)),
            actual_seq_k=[n_pools] * rows,
            block_table=torch.arange(rows, dtype=torch.int32).view(rows, 1),
        )
    elif layout != "BSND":
        raise ValueError(layout)
    # Independent literal ranking: scores increase strictly with pool id.
    selected = torch.arange(n_pools - topk // pool_size, n_pools, dtype=torch.int32)
    tokens = (selected[:, None] * pool_size + torch.arange(pool_size)).flatten()
    tail = torch.arange(n_pools * pool_size, n_pools * pool_size + 3, dtype=torch.int32)
    expected = torch.cat((tokens, tail)).to(torch.int32).expand(rows, -1)
    if layout == "BSND":
        expected = expected.unsqueeze(1)
    return inputs, expected


def _binding():
    # Missing wheel, schema, vendor library or kernel is a failure, not a skip.
    import attentions  # noqa: F401

    return torch.ops.attentions.pool_key_indexer


# Six representative tuples replace layout x dtype x shape expansion.
@pytest.mark.parametrize(
    "layout,dtype,rows,n_pools,heads,topk",
    [
        pytest.param(
            "TND", torch.bfloat16, 16, 512, 32, 2048, id="W-glm53-flash-tnd-topk2048"
        ),
        pytest.param(
            "BSND", torch.bfloat16, 16, 512, 32, 2048, id="W-glm53-flash-bsnd-topk2048"
        ),
        pytest.param("TND", torch.float16, 1, 16, 1, 16, id="edge-tnd-fp16-minimum"),
        pytest.param("BSND", torch.float16, 1, 16, 1, 16, id="edge-bsnd-fp16-minimum"),
        pytest.param(
            "TND", torch.bfloat16, 3, 32, 4, 16, id="edge-odd-requests-ranked-subset"
        ),
        pytest.param(
            "BSND", torch.bfloat16, 16, 64, 32, 16, id="edge-multi-head-ranked-subset"
        ),
    ],
)
def test_pool_key_indexer(layout, dtype, rows, n_pools, heads, topk):
    operator = _binding()
    inputs, expected = generate_test_data(layout, rows, n_pools, heads, dtype, topk)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        actual = operator(**inputs)
        # Pool order is not assumed. Check the selected token set and the
        # separate ordered tail, rather than accepting arbitrary in-range ids.
        assert actual.shape == expected.shape
        assert actual.dtype == torch.int32
        topk = inputs["topk"]
        assert_accuracy(actual[..., :topk].sort().values, expected[..., :topk])
        assert_accuracy(actual[..., topk:], expected[..., topk:])
    executor = partial(operator, **inputs)
    executor.impl = "aclnn"
    executor.impl_ref = "independent_ranked_fixture"
    executor.accuracy_checked = accuracy_checked
    _check_binding_performance(
        executor,
        enable_perf=(rows, n_pools, heads) in perf_shape
        and layout == "TND"
        and dtype == torch.bfloat16,
    )


@pytest.mark.parametrize(
    "change,error",
    [
        ({"quant_mode": 0}, "quant_mode=-1"),
        ({"return_value": True}, "return_value=true"),
        ({"pool_size": 0}, "pool_size must be > 0"),
        ({"topk": 1}, "must be >= pool_size"),
        ({"layout_q": "BAD"}, "unsupported layout_q"),
        ({"layout_k": "BAD"}, "unsupported layout_k"),
        ({"query_dtype": torch.float32}, "query must be BFloat16 or Half"),
        ({"key_dtype": torch.float16}, "pool_key dtype must match query"),
        ({"weights_dtype": torch.float16}, "weights dtype must match query"),
        ({"query_rank": 1}, "query must be"),
    ],
)
def test_pool_key_indexer_validation(change, error):
    operator = _binding()
    inputs, _ = generate_test_data("BSND", 1, 16, 1, torch.bfloat16)
    change = dict(change)
    for key, tensor in (
        ("query_dtype", "query"),
        ("key_dtype", "pool_key"),
        ("weights_dtype", "weights"),
    ):
        if key in change:
            inputs[tensor] = inputs[tensor].to(change.pop(key))
    if "query_rank" in change:
        inputs["query"] = inputs["query"].flatten()
        change.pop("query_rank")
    inputs.update(change)
    with pytest.raises(RuntimeError, match=error):
        operator(**inputs)
