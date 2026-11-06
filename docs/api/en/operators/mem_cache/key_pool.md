# key_pool

Public import: `from sgl_kernel_npu.operators.mem_cache import key_pool`.
`auto` and `aclnn` use the separate `attentions` extension and CANN's
`aclnnKeyPoolGetWorkspaceSize` / `aclnnKeyPool`. There is no fallback or
`cann_ops_transformer` package dependency.

```python
pooled = key_pool(hidden_states, wk, gate_weight, ape, state_cache,
                  cache_block_table, start_pos,
                  norm_weight=norm_weight, norm_bias=norm_bias,
                  cu_seqlens=cu_seqlens, cmp_ratio=4, norm_eps=1e-6)
```

- Hidden states: contiguous ND BF16/FP16 `[T,H]` or `[B,S,H]`.
  Weights: same dtype `[D,H]`; APE: FP32 `[r,D]`.
- Mutable state: FP32 `[blocks,block_size,2D]`, normalized Key followed by
  Gate; only the first axis may have padding/stride. The Torch schema marks
  `state_cache` as `Tensor(a!)`. APE is not stored in state.
- Table: device int32 `[B,L]`; starts: device int32 `[B]`. This table maps
  temporary projection state, not the final compressed index cache.
- TH requires device int32 `cu_seqlens[B+1]`, beginning at zero and ending
  at T. BSH forbids it. Length values stay on device through graph replay.
- Norm weight/bias: paired FP32 `[D]`. `cos`, `sin`, `seqused` must be None.
  Compression ratios are powers of two from 2 to 128. No RoPE is implemented.
- One Tensor is returned. TH capacity is `min(T,T//r+B)`, with valid rows
  compacted in request order. Each request contributes
  `(start+Q)//r-start//r` rows. Other rows are uninitialized. BSH output is
  `[B,ceil(S/r),D]`, with the valid prefix in each request.

Build/install both Python facades and the updated extension with the serving
Python environment. The main sgl_kernel_npu wheel alone does not supply attentions:

```bash
bash build.sh -a attentions
python3 -m pip install --no-deps --force-reinstall output/attentions-*.whl
python3 -c 'import torch, torch_npu, attentions; print(torch.ops.attentions.key_pool.default._schema)'
python3 -m pytest -q tests/python/operators/mem_cache/test_key_pool.py
```

The CANN/custom-op installation must actually export both KeyPool symbols.
NPUGraph tests and independent projection/LN/APE/state references are provided;
passing import/schema checks does not establish numerical or graph support.
TorchAir converter support is not provided by the upstream 9.2.0 interface.

Contract source: [ops-transformer 9.2.0](https://gitcode.com/cann/ops-transformer/blob/9.2.0/attention/key_pool/docs/torchapi_key_pool.md).
