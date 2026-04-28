import ast
import inspect
from pathlib import Path

import sgl_kernel_npu.fla.chunk as chunk
import sgl_kernel_npu.fla.chunk_delta_h as chunk_delta_h
import sgl_kernel_npu.fla.chunk_o as chunk_o
import sgl_kernel_npu.fla.chunk_scaled_dot_kkt as chunk_scaled_dot_kkt
import sgl_kernel_npu.fla.cumsum as cumsum
import sgl_kernel_npu.fla.solve_tril as solve_tril
import sgl_kernel_npu.fla.wy_fast as wy_fast


def _param_names(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


def test_public_wrappers_accept_prebuilt_metadata_kwargs():
    assert "prebuilt_meta" in _param_names(chunk.chunk_gated_delta_rule_npu)
    assert "prebuilt_meta" in _param_names(chunk.chunk_gated_delta_rule_fwd)
    assert "block_indices" in _param_names(cumsum.chunk_local_cumsum)
    assert "block_indices" in _param_names(cumsum.chunk_local_cumsum_scalar_npu)
    assert "chunk_indices" in _param_names(chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_npu)
    assert "chunk_indices_large_block" in _param_names(solve_tril.solve_tril_npu)
    assert "chunk_indices_bt" in _param_names(solve_tril.solve_tril_npu)
    assert "chunk_indices" in _param_names(wy_fast.recompute_w_u_fwd_npu)
    assert "chunk_indices" in _param_names(chunk_delta_h.chunk_gated_delta_rule_fwd_h_npu)
    assert "chunk_offsets" in _param_names(chunk_delta_h.chunk_gated_delta_rule_fwd_h_npu)
    assert "chunk_indices" in _param_names(chunk_o.chunk_fwd_o_npu)
    assert "chunk_offsets" in _param_names(chunk_o.chunk_fwd_o_npu)


def test_chunk_fwd_passes_all_prebuilt_meta_fields():
    source = Path(chunk.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    keyword_names = {
        keyword.arg
        for call in calls
        for keyword in call.keywords
        if keyword.arg is not None
    }
    assert "block_indices" in keyword_names
    assert "chunk_indices" in keyword_names
    assert "chunk_offsets" in keyword_names
    assert "chunk_indices_large_block" in keyword_names
    assert "chunk_indices_bt" in keyword_names
