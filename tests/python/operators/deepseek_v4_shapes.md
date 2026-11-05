# DeepSeek-V4-Flash operator test shapes

The primary cases use dimensions from the official
[DeepSeek-V4-Flash configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/fd53f944496234770ba80e15004f9b6d269a71f5/config.json).
They use generated values, not captured production activations. Token counts and
sequence lengths are representative decode/prefill workloads; they are not fixed
model dimensions. The PR launch scripts select `DeepSeek-V4-Flash-0731` / its ModelSlim
quantized variant. Their page size and TP/DP settings are accounted for below;
the model dimensions are traced to the official configuration and SGLang call
sites rather than inferred from the checkpoint directory name.

| Model field | Value |
| --- | ---: |
| `hidden_size` | 4096 |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 20 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 1 |
| `head_dim` / `qk_rope_head_dim` | 512 / 64 |
| `sliding_window` | 128 |
| `index_n_heads` / `index_head_dim` / `index_topk` | 64 / 128 / 512 |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 6 |
| `scoring_func` / `routed_scaling_factor` | sqrtsoftplus / 1.5 |
| Compressed attention ratios | 4 and 128 |

## Operator inputs

`T` is the number of input tokens. `P` is the number of allocated cache pages.
The examples assume 128-token original KV pages, as in the PR's deployment
scripts. C4 cache pages contain 128 / 4 = 32 compressed tokens.

| Operator | Primary input geometry | Primary cases |
| --- | --- | --- |
| `sparse_flash_mla` and metadata | Q `[T,64,512]`; BF16 KV `[P,128,1,512]`; window left/right 127/0 | T=1 decode and T=128 prefill; original KV length 256 |
| `mixed_quant_sparse_flash_mla` and metadata | Q `[T,64,512]`; packed KV `[P,128,1,640]` on 9.1 or `[P,128,1,608]` on 9.2 | T=1 decode and T=128 prefill; original KV length 128 |
| `quant_lightning_indexer` and metadata | Q `[T,64,128]`; K `[P,32,1,128]`; top-k 512; C4 | T=1 decode and T=128 prefill; compressed KV length 513 |
| `compressor` | X `[128,4096]`; each W `[2D,4096]`; APE `[4,2D]`; C4 overlap coefficient 2 | Attention D=512 and indexer D=128 |
| `kv_compress_epilog` | X `[T,512]`; NoPE 448 and RoPE 64; 64-element quant groups | T=1 decode and T=128 page |
| `indexer_quant_cache` | X `[T,128]`; FP8 keys and FP32 scales | T=1 and T=8; normal FP8 mode |
| `mhc_pre` | X `[T,4,4096]`; mixing weight `[24,16384]`; scale `[3]`; base `[24]` | T=1 and T=16 |
| `mhc_post` | X `[T,4096]`; residual `[T,4,4096]`; post `[T,4]`; comb `[T,4,4]` | T=1 and T=16 |
| `inplace_partial_rotary_mul` | X `[T,1,64,512]`; partial slice `[448,512]` | T=1 and T=16 |
| `moe_gating_top_k` | Router logits `[T,256]`; bias `[256]`; k=6; norm_type=2; renorm=0; scale=1.5 | T=1 and T=16 |

The SGLang NPU call sites are the
[attention backend](https://github.com/sgl-project/sglang/pull/47879/files),
`models/deepseek_v4.py`, `layers/attention/dsv4/compressor.py`,
`layers/attention/dsv4/indexer.py`, and `hardware_backend/npu/moe/topk.py`.
Compressor projection input is `hidden_size=4096`, not `q_lora_rank=1024`.
The NPU indexer replicates its 64 heads. Attention heads are divided by attention
TP, which is one when attention DP equals TP in the supplied launch scripts;
the primary attention cases therefore use 64 heads. Other sharding arrangements
require their corresponding per-rank shape.

CANN 9.1 A3 does not support the deployed MoE kernel's `norm_type=2` path, so the
model-mode case explicitly skips there. Supplementary softmax/sigmoid cases do
not establish support for DeepSeek-V4-Flash's sqrtsoftplus routing. The indexer
cache test exercises the FP8 writer API; the SGLang A3 INT8 cache path is a
different path.

## IDs and test organization

`pytest.param(..., id="deepseek-v4-flash-decode-t1-e256-k6")` gives the parameter
set a readable name in pytest output and permits selection with `-k`.
It does not change tensor data or select a model at runtime. Model-derived
dimensions carry the model name; artificial modes and boundary cases retain a
`synthetic-` ID and use the existing `extra_checks` marker.

Each migrated operator has two primary parameter sets. Additional regression
coverage is retained under `extra_checks`; select it explicitly with
`-m extra_checks`, or use `-m ''` to collect primary and supplementary cases.

Tests generate data, compare independent results, and only then profile.
Non-mutating mHC/MoE tests use the shared `check_accuracy` helper directly.
In-place writers/RoPE and compressor require independent buffers; metadata
requires a separately constructed reference descriptor, and indexer compares
top-k sets. Those tests use the existing `assert_accuracy` and `make_executor`
helpers locally, keeping these operator-specific rules outside the shared
accuracy framework. Metadata is timed separately from its paired compute;
binary tiling bytes are never compared to an eager descriptor.

Changing a shape or label does not constitute a new NPU accuracy result. Run
the primary cases on each target hardware/SDK combination before reporting
updated device accuracy or performance.
