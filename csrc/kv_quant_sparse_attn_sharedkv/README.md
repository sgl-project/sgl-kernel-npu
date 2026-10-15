# Quantized sparse shared-KV attention

This directory ports the Ascend 950 operator updated in
[`vllm-ascend@683b6c009d`](https://github.com/vllm-project/vllm-ascend/commit/683b6c009d)
as `torch.ops.npu.kv_quant_sparse_attn_sharedkv`. It is a separate A5-only
operator; the A3 `sparse_attn_sharedkv` directory and its metadata format are
unchanged. The source commit adds per-query original-KV sparse indices and
generalizes the original-KV window on the A5 path.

The call returns `(attention_out, softmax_lse)`. `attention_out` has the same
shape and BF16 dtype as `q`; `softmax_lse` is currently an empty FP32 tensor.
For PageAttention, `q` is contiguous BF16 `[B, S, 64|128, 512]` (`BSND`) or
`[T, 64|128, 512]` (`TND`). `ori_kv` and optional `cmp_kv` are contiguous
FP8 E4M3FN packed caches `[num_blocks, block_size, 1, 640]`, with block sizes
divisible by 16 and at most 1024. The 640-byte packed row contains 64 BF16
RoPE values, 448 FP8 NoPE values, seven E8M0 scale bytes, and padding; an
ordinary FP8 tensor of 640 independent values is not a valid cache. The query
head count is 64 or 128 and the KV head count is one. The operator requires
`kv_quant_mode=1`, `tile_size=64`, `rope_head_dim=64`, `ori_mask_mode=4`,
`cmp_mask_mode=3`, `layout_kv="PA_ND"`, and FP32 `sinks`. The default
`cmp_ratio=1` is for original-KV-only sliding-window attention; compressed-KV
attention uses 4 or 128. `return_softmax_lse=True` is not supported.

`ori_sparse_indices` is the feature introduced by the source commit. In this
port, it is accepted for TND original-KV-only attention as contiguous INT32
`[T, 1, K]`: each row lists **absolute physical cache slots**
(`block_id * block_size + offset`), with valid slots first and `-1` padding
afterward. Use `K=512` or `1024`. It cannot be combined with `cmp_kv` in this
binding. `ori_win_left` and `ori_win_right` may be nonnegative INT32 values;
the older upstream README's fixed-window description predates this commit.

The kernel requires `metadata` as NPU INT32 `[1024]`. Build it with the
separate `torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host` op,
passing contiguous **CPU** INT32 `cu_seqlens_q` for TND and/or `seqused_kv`.
Its A5 task table has 36 cube-core slots of 9 integers followed by 72
vector-core slots of 8 integers. Do not pass metadata from the existing
`sparse_attn_sharedkv_metadata_host`, whose cube-core stride is 8.
`ori_block_table`, `seqused_kv`, `sinks`, and `metadata` are required. TND
also requires `cu_seqlens_q`. The `cu_seqlens_ori_kv` and
`cu_seqlens_cmp_kv` placeholders remain unsupported.

On an Ascend 950 machine, build the A5 kernel module and run the focused
tests with:

```bash
./build.sh -a kernels Ascend950PR_9599
python -m pip install --force-reinstall --no-deps output/sgl_kernel_npu*.whl
python tests/python/sgl_kernel_npu/test_kv_quant_sparse_attn_sharedkv_metadata.py
python tests/python/sgl_kernel_npu/test_kv_quant_sparse_attn_sharedkv.py
```

The main test launches the TND sparse-index path with an all-zero packed KV
cache and checks the BF16 result. A complete numerical comparison for
nonzero packed FP8 KV still requires an Ascend 950 test fixture.
