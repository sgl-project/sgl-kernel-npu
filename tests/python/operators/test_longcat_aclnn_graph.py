"""Exercise the migrated native launch path with changing replay inputs."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.operators.embedding.test_compute_n_gram_ids import (
    generate_data as generate_ngram_data,
)
from tests.python.operators.indexer.test_mlp_lightning_indexer import _assert_selection
from tests.python.operators.indexer.test_mlp_lightning_indexer import (
    generate_data as generate_indexer_data,
)
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl


def _capture(call):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        result = call()
    torch.npu.synchronize()
    return graph, result


def _native_impl(operator):
    impl, _ = get_test_impls()
    if impl not in ("auto", "ascendc"):
        pytest.skip("This regression exercises the native ACLNN launch path")
    require_test_impl(operator, impl)
    return impl


@pytest.mark.parametrize("ignore", [[], [9]])
def test_oe_aclnn_graph_replay(ignore):
    from sgl_kernel_npu.operators.embedding import (
        compute_n_gram_ids,
        update_oe_token_table,
    )

    impl = _native_impl(compute_n_gram_ids)
    _native_impl(update_oe_token_table)
    inputs = generate_ngram_data(3, 2, 4, 5, 4, "normal")
    initial = inputs["oe_token_table"].clone()
    update_inputs = {
        name: inputs[name]
        for name in (
            "tokens",
            "row_indices",
            "column_starts",
            "batch_size",
            "max_context_len",
        )
    }
    update_inputs["req_lens"] = torch.full((3,), 2, dtype=torch.int32)
    update_inputs["ignore_tokens"] = torch.tensor(ignore, dtype=torch.int32)

    def call():
        table = update_oe_token_table(
            **update_inputs, oe_token_table=inputs["oe_token_table"], impl=impl
        )
        return table, compute_n_gram_ids(**inputs, impl=impl)

    graph, (actual_table, actual_ids) = _capture(call)
    assert actual_table.data_ptr() == inputs["oe_token_table"].data_ptr()
    for value in (9, 17, 25):
        inputs["tokens"].fill_(value)
        expected_table = initial.clone()
        update_oe_token_table(
            **update_inputs, oe_token_table=expected_table, impl="torch_native"
        )
        expected_ids = compute_n_gram_ids(
            **{**inputs, "oe_token_table": expected_table}, impl="torch_native"
        )
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert_accuracy(actual_table, expected_table)
        assert_accuracy(actual_ids, expected_ids)


@pytest.mark.parametrize("return_value", [False, True])
def test_indexer_aclnn_graph_replay(return_value):
    from sgl_kernel_npu.operators.indexer import mlp_lightning_indexer

    impl = _native_impl(mlp_lightning_indexer)
    inputs = generate_indexer_data(
        dict(q_lens=[1, 2], k_lens=[32, 64], return_value=return_value),
        torch.bfloat16,
    )
    graph, actual = _capture(lambda: mlp_lightning_indexer(**inputs, impl=impl))
    for _ in range(3):
        inputs["query"].neg_()
        reference = mlp_lightning_indexer(**inputs, impl="torch_native")
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        _assert_selection(actual, reference, inputs)
