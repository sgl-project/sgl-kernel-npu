"""TorchNPU attention/indexer forwarding through the production registry."""

import sys
from unittest.mock import Mock, sentinel

import pytest

_PAGED = dict(
    query=sentinel.query,
    key_cache=sentinel.key,
    num_heads=8,
    num_kv_heads=2,
    scale_value=0.25,
    block_table=sentinel.pages,
    context_lens=sentinel.lengths,
    out=sentinel.out,
)
_QKV = (sentinel.query, sentinel.key, sentinel.value)
_MLA_ARGS = (
    sentinel.tokens,
    sentinel.dq,
    sentinel.uq_qr,
    sentinel.uk,
    sentinel.dkv_kr,
    sentinel.gamma_cq,
    sentinel.gamma_ckv,
    sentinel.sin,
    sentinel.cos,
    sentinel.kv_cache,
    sentinel.kr_cache,
)


@pytest.fixture
def torch_npu_stub(monkeypatch):
    stub = Mock()
    monkeypatch.setitem(sys.modules, "torch_npu", stub)
    return stub


@pytest.mark.parametrize("explicit_impl", [False, True], ids=["auto", "explicit"])
@pytest.mark.parametrize(
    "category,op_name,symbol,args,kwargs",
    [
        pytest.param(
            "attention",
            "flash_attention_qlens",
            "_npu_flash_attention_qlens",
            (),
            dict(
                _PAGED, value_cache=sentinel.value, mask=sentinel.mask, seq_len=[2, 4]
            ),
            id="paged-prefill-out",
        ),
        pytest.param(
            "attention",
            "paged_attention",
            "_npu_paged_attention",
            (),
            dict(_PAGED, value_cache=sentinel.value, workspace=sentinel.workspace),
            id="paged-decode-out-workspace",
        ),
        pytest.param(
            "attention",
            "paged_attention_mla",
            "_npu_paged_attention_mla",
            (),
            dict(_PAGED, mla_vheadsize=512),
            id="paged-mla-decode-out",
        ),
        pytest.param(
            "attention",
            "attention_update",
            "npu_attention_update",
            ((sentinel.lse0, sentinel.lse1), (sentinel.part0, sentinel.part1), 0),
            {},
            id="dcp-merge",
        ),
        pytest.param(
            "attention",
            "fused_infer_attention_score",
            "npu_fused_infer_attention_score",
            _QKV,
            dict(
                actual_seq_lengths=[2, 4],
                actual_seq_lengths_kv=sentinel.lengths,
                block_table=sentinel.pages,
                query_rope=sentinel.query_rope,
                key_rope=sentinel.key_rope,
                antiquant_mode=1,
            ),
            id="fia",
        ),
        pytest.param(
            "attention",
            "fused_infer_attention_score_out",
            "npu_fused_infer_attention_score.out",
            _QKV,
            dict(workspace=sentinel.workspace, out=[sentinel.out, sentinel.lse]),
            id="fia-out-workspace",
        ),
        pytest.param(
            "attention",
            "fused_infer_attention_score_get_max_workspace",
            "_npu_fused_infer_attention_score_get_max_workspace",
            _QKV,
            dict(input_layout="TND", num_heads=8, block_table=sentinel.pages),
            id="fia-workspace",
        ),
        pytest.param(
            "attention",
            "fused_infer_attention_score_v2",
            "npu_fused_infer_attention_score_v2",
            _QKV,
            dict(
                num_query_heads=8,
                softmax_scale=0.125,
                actual_seq_qlen=[2],
                learnable_sink=sentinel.sink,
                return_softmax_lse=True,
            ),
            id="fia-v2",
        ),
        pytest.param(
            "attention",
            "fused_infer_attention_score_v2_out",
            "npu_fused_infer_attention_score_v2.out",
            _QKV,
            dict(workspace=sentinel.workspace, out=[sentinel.out, sentinel.lse]),
            id="fia-v2-out-workspace",
        ),
        pytest.param(
            "attention",
            "fused_infer_attention_score_v2_get_max_workspace",
            "_npu_fused_infer_attention_score_v2_get_max_workspace",
            _QKV,
            dict(input_layout="TND", num_query_heads=8, block_table=sentinel.pages),
            id="fia-v2-workspace",
        ),
        pytest.param(
            "attention",
            "kv_quant_sparse_flash_attention_torch_npu",
            "npu_kv_quant_sparse_flash_attention",
            (*_QKV, sentinel.indices, 0.25, 2, 2),
            dict(layout_query="TND", layout_kv="PA_BSND", block_table=sentinel.pages),
            id="torch-npu-quant-sparse-kv",
        ),
        pytest.param(
            "indexer",
            "lightning_indexer",
            "npu_lightning_indexer",
            (sentinel.query, sentinel.key, sentinel.weights),
            dict(layout_query="TND", layout_key="PA_BSND", sparse_count=2048),
            id="lightning-indexer",
        ),
        pytest.param(
            "attention",
            "mla_prolog_v3",
            "npu_mla_prolog_v3",
            _MLA_ARGS,
            dict(cache_index=sentinel.cache_index, query_quant_mode=1),
            id="mla-prolog-v3-mutates-cache",
        ),
    ],
)
def test_torch_npu_attention_dispatch(
    registry_for_generation,
    torch_npu_stub,
    category,
    op_name,
    symbol,
    args,
    kwargs,
    explicit_impl,
):
    operator_modules = registry_for_generation("ARCH22")
    api = operator_modules.import_module(f"operators.{category}")
    call = getattr(api, op_name)
    registry = operator_modules.registry
    assert op_name in api.__all__
    assert registry.get_op_name(call) == f"{category}.{op_name}"
    legacy_atb_ops = {"flash_attention_qlens", "paged_attention", "paged_attention_mla"}
    vendor_impl = (
        "ascend-transformer-boost" if op_name in legacy_atb_ops else "ops-transformer"
    )
    implementations = ["auto", vendor_impl]
    if not op_name.endswith("_get_max_workspace"):
        implementations.append("torch_native")
    assert registry.list_impls(call) == sorted(implementations)
    assert registry.get_impl(call, "auto") is registry.get_impl(call, vendor_impl)

    sdk = torch_npu_stub
    for name in symbol.split("."):
        sdk = getattr(sdk, name)
    sdk.return_value = sentinel.result

    selection = {"impl": vendor_impl} if explicit_impl else {}
    assert call(*args, **kwargs, **selection) is sentinel.result
    sdk.assert_called_once_with(*args, **kwargs)


def test_fia_unknown_impl_and_sdk_error(operator_modules, torch_npu_stub):
    api = operator_modules.import_module("operators.attention")
    sdk = torch_npu_stub.npu_fused_infer_attention_score
    with pytest.raises(operator_modules.registry.ImplementationNotFoundError):
        api.fused_infer_attention_score(*_QKV, impl="unregistered")
    sdk.assert_not_called()

    sdk.side_effect = RuntimeError("unsupported by the installed SDK")
    with pytest.raises(RuntimeError, match="unsupported by the installed SDK"):
        api.fused_infer_attention_score(*_QKV)
    sdk.assert_called_once_with(*_QKV)


@pytest.mark.parametrize(
    "name", ["flash_attention_qlens", "paged_attention", "paged_attention_mla"]
)
def test_legacy_atb_vendor_is_arch22_only(registry_for_generation, name):
    modules = registry_for_generation("ARCH35")
    operator = getattr(modules.import_module("operators.attention"), name)
    for impl in ("auto", "ascend-transformer-boost"):
        with pytest.raises(modules.registry.UnsupportedGenerationError):
            modules.registry.get_impl(operator, impl)
    assert callable(modules.registry.get_impl(operator, "torch_native"))
