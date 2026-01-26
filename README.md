# SGLang-Kernel-NPU


## Introduction

**SGLang-Kernel-NPU** is the official kernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for Ascend NPU. It delivers high-performance, production-ready compute primitives optimized for large language model (LLM) inference on Ascend hardware.

The library consists of two main components:
- **DeepEP-Ascend**: Ascend implementation of [DeepEP](https://github.com/deepseek-ai/DeepEP), providing highly optimized Expert Parallelism (EP) communication kernels for Mixture-of-Experts (MoE) models.
- **SGLang-Kernel-NPU**: A comprehensive collection of optimized inference kernels including attention mechanisms, normalization, activation functions, LoRA adapters, and more.

For contribution guidelines, please refer to the [Contribution Guide](docs/developer_guide/contribution_guide.md).


## Features

### DeepEP-Ascend

DeepEP-Ascend provides optimized all-to-all communication kernels for Expert Parallelism in MoE models.

**Communication Modes:**
- **Normal Mode**: High-throughput dispatch and combine operations for training and prefill phases (up to 4096 tokens/batch)
- **Low-Latency Mode**: Optimized for production inference with small batch sizes (128 tokens/batch), achieving sub-150us latency

**Key Capabilities:**
- Token dispatch and combine with automatic load balancing
- Fused MoE computation (`fused_deep_moe`)
- Intranode HCCS and internode RDMA communication
- INT8/FP8/BF16 quantization for reduced memory bandwidth
- Support for EP scales: 2, 4, 8, 16, 32, 64, 128, 144, 160 ranks

### SGLang-Kernel-NPU

SGLang-Kernel-NPU provides a comprehensive set of optimized inference kernels:

**Attention:**
- Multi-Latent Attention (MLA) with Paged KV Cache support
- Grouped Query Attention (GQA)
- Decode Attention with optimized memory access patterns

**Flash Linear Attention (FLA):**
- Gated Delta Rule implementation
- Chunk-based operations for efficient memory usage

**Normalization:**
- RMSNorm
- Fused Add + RMSNorm + Bias
- Split QKV + RMSNorm + RoPE fusion

**Activation Functions:**
- SwiGLU
- Quantized SwiGLU (INT8)

**LoRA Adapters:**
- BGMV expand/shrink
- SGMV expand/shrink
- SGEMMV expand/shrink

**Speculative Decoding:**
- Efficient tree building
- Greedy tree verification

**MLA Preprocessing:**
- End-to-end fusion: RMSNorm → Dequant → MatMul → RoPE → ReshapeAndCache

**Mamba Support:**
- Causal Conv1D for state space models (SSM)

**KV Cache Management:**
- Paged Attention support
- Cache location assignment and update

**Other Utilities:**
- Lightning Indexer for sparse Top-K indexing
- Triangular matrix inverse
- Batch MatMul with transpose


## Quick Start

DeepEP-Ascend: Ascend Implementation of DeepEP. [README](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)

SGLang-Kernel-NPU: Other SGLang Kernels for Ascend NPU. [README](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)


## DeepEP-Ascend Performance

### Normal Kernels with Pure HCCS

We test normal kernels on A3 384 SuperPOD, following the DeepSeek-V3/R1 pretraining setting (4096 tokens per batch, 7168 hidden size, top-8 experts, INT8 dispatching and BF16 combining).

| Type      | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
| --------- | ------------ | -------------------- | ----------- | -------------------- |
| Intranode | 8            | 146 GB/s (HCCS)      | 8           | 125 GB/s (HCCS)      |
| Intranode | 16           | 107 GB/s (HCCS)      | 16          | 103 GB/s (HCCS)      |
| Intranode | 32           | 102 GB/s (HCCS)      | 32          | 95 GB/s (HCCS)       |
| Intranode | 64           | 81 GB/s (HCCS)       | 64          | 91 GB/s (HCCS)       |
| Intranode | 128          | 57 GB/s (HCCS)       | 128         | 81 GB/s (HCCS)       |

### Low-Latency Kernels with Pure HCCS

We test low-latency kernels on A3 384 SuperPOD, following a typical DeepSeek-V3/R1 production setting (128 tokens per batch, 7168 hidden size, top-8 experts, INT8 dispatching and BF16 combining).

| Dispatch #EP | Latency | Bandwidth      | Combine #EP | Latency | Bandwidth       |
| ------------ | ------- | -------------- | ----------- | ------- | --------------- |
| 8            | 132 us  | 58 GB/s (HCCS) | 8           | 126 us  | 116 GB/s (HCCS) |
| 16           | 139 us  | 55 GB/s (HCCS) | 16          | 135 us  | 109 GB/s (HCCS) |
| 32           | 153 us  | 49 GB/s (HCCS) | 32          | 151 us  | 97 GB/s (HCCS)  |
