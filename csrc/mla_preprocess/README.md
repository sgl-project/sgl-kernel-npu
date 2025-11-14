## Function
Fuses the entire process of PagedAttention input data processing in MLA scenarios, including a series of computations starting from hidden state input through RMSNorm, dequantization, matrix multiplication, RoPE, and reshapeAndCache.

## Input and Output List
### Input Tensor
| Module Function      | Identifier | Name          | Data Type            | Data Format | Dimension                                      | Detailed Description                                                                 |
|----------------------|------------|---------------|----------------------|-------------|-----------------------------------------------|-------------------------------------------------------------------------------------|
| Input Data           | inTensor0  | hiddenState   | float16/bf16         | ND          | [tokenNum, hiddenSize], hiddenSize values: [2048,8192] | Mandatory.                                                                          |
| rmsNormQuant_0       | inTensor1  | gamma0        | float16/bf16         | ND          | [hiddenSize], hiddenSize values: [2048,8192]  | Mandatory. Data type consistent with input.                                         |
|                      | inTensor2  | beta0         | float16/bf16         | ND          | [hiddenSize], hiddenSize values: [2048,8192]  | Mandatory. Data type consistent with input.                                         |
|                      | inTensor3  | quant_scale0  | float16/bf16         | ND          | [1]                                           | Mandatory, supports empty tensor. Only passed when quant_mode is 0, data type consistent with input. |
|                      | inTensor4  | quant_offset0 | int8                 | ND          | [1]                                           | Mandatory, supports empty tensor. Only passed when quant_mode is 0.                  |
| matmul_0             | inTensor5  | wdqkv         | int8<br>float16/bf16 | NZ          | [1,224,2112,32]                               | Mandatory.<br>When data type matches input, RMSNormQuant is not enabled.             |
|                      | inTensor6  | descale0      | int64/float          | ND          | [2112]                                        | Mandatory. int64 when input is fp16, float when input is bf16.                      |
|                      | inTensor7  | bias0         | int32                | ND          | [2112]                                        | Mandatory, supports empty tensor. Not passed when quant_mode is 1 or 3.             |
| rmsNormQuant_1       | inTensor8  | gamma1        | float16/bf16         | ND          | [1536]                                        | Mandatory. Data type consistent with input.                                         |
|                      | inTensor9  | beta1         | float16/bf16         | ND          | [1536]                                        | Mandatory. Data type consistent with input.                                         |
|                      | inTensor10 | quant_scale1  | float16/bf16         | ND          | [1]                                           | Mandatory, supports empty tensor. Only passed when quant_mode is 0, data type consistent with input. |
|                      | inTensor11 | quant_offset1 | int8                 | ND          | [1]                                           | Mandatory, supports empty tensor. Only passed when quant_mode is 0.                  |
| matmul_1             | inTensor12 | wuq           | int8<br>float16/bf16 | NZ          | [1,48,headNum*192,32]                         | Mandatory.<br>When data type matches input, RMSNormQuant is not enabled.             |
|                      | inTensor13 | descale1      | int64/float          | ND          | [headNum*192]                                 | Mandatory. int64 when input is fp16, float when input is bf16.                      |
|                      | inTensor14 | bias1         | int32                | ND          | [headNum*192]                                 | Mandatory, supports empty tensor. Not passed when quant_mode is 1 or 3.             |
| rmsNorm              | inTensor15 | gamma2        | float16/bf16         | ND          | [512]                                         | Mandatory. Data type consistent with input.                                         |
| rope                 | inTensor16 | cos           | float16/bf16         | ND          | [tokenNum,64]                                | Mandatory. Data type consistent with input.                                         |
|                      | inTensor17 | sin           | float16/bf16         | ND          | [tokenNum,64]                                | Mandatory. Data type consistent with input.                                         |
| matmulEin            | inTensor18 | wuk           | float16/bf16         | ND/NZ       | ND:[headNum,128,512]<br>NZ:[headNum,32,128,16] | Mandatory. Data type consistent with input.                                         |
| reshapeAndCache      | inTensor19 | kv_cache      | float16/bf16/int8    | ND/NZ       | cache_mode=0：<br>[blockNum,blockSize,1,576]<br>cache_mode=1：<br>[blockNum,blockSize,1,512]<br>cache_mode=2：<br>[blockNum, 1*512/32, block_size, 32]<br>cache_mode=3：<br>[blockNum, 1*512/16, block_size, 16] | Mandatory. Data type consistent with input.<br>When cache_mode=1, tensor shape is split. When cache_mode=2, format is NZ, type is int8. When cache_mode=3, format is NZ. |
|                      | inTensor20 | kv_cache_rope | float16/bf16         | ND/NZ       | cache_mode=1：<br>[blockNum,blockSize,1,64]<br>cache_mode=2 or 3：<br>[blockNum, headNum*64/16, block_size, 16] | Mandatory, supports empty tensor. Passed when cacheMode≠0, data type consistent with input. When cache_mode=2 or 3, format is NZ. |
|                      | inTensor21 | slotmapping   | int32                | ND          | [tokenNum]                                   | Mandatory.                                                                          |
| quant                | inTensor22 | ctkv_scale    | float16/bf16         | ND          | [1]                                           | Mandatory, supports empty tensor. Passed when cache_mode=2, data type consistent with input. |
|                      | inTensor23 | q_nope_scale  | float16/bf16         | ND          | [headNum]                                    | Mandatory, supports empty tensor. Passed when cache_mode=2, data type consistent with input. |

### cache_mode
- When cache_mode=1, input/output kcache is split into krope and ctkv, q is split into qrope and qnope.
- When cache_mode=2, based on cache_mode=1, krope and ctkv are converted to NZ format output, ctkv and qnope are statically symmetrically quantized per_head to int8 type.
- When cache_mode=3, based on cache_mode=1, krope and ctkv are converted to NZ format output.

### quant_mode
- PER_TENSOR_QUANT_ASYMM: per_tensor static asymmetric quantization, default quantization type;
- PER_TOKEN_QUANT_SYMM: per_token dynamic symmetric quantization;

## Output Tensor
Optional output tensors cannot use empty tensors as placeholders
| Module Function | Identifier | Name           | Data Type          | Data Format | Dimension                                                                 | Detailed Description                                                                                |
|-----------------|------------|----------------|--------------------|------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Output Data     | outTensor0 | q_out0         | float16/bf16/int8  | ND         | cache_mode=0：<br>[tokenNum,headNum,576]<br>cache_mode=1/2/3：<br>[tokenNum,headNum,512] | Output tensor. Data type consistent with input. When cache_mode=2, data type is int8.                |
|                 | outTensor1 | kv_cache_out0  | float16/bf16/int8  | ND/NZ      | cache_mode=0：<br>[blockNum,blockSize,1,576]<br>cache_mode=1：<br>[blockNum,blockSize,1,512]<br>cache_mode=2：<br>[blockNum, headNum*512/32,block_size, 32]<br>cache_mode=3：<br>[blockNum, headNum*512/16,block_size, 16] | Output tensor. Data type consistent with input. When cache_mode=2, data type is int8, format is NZ. When cache_mode=3, format is NZ. Same tensor as input kvCache. |
|                 | outTensor2 | q_out1         | float16/bf16       | ND         | [tokenNum,headNum,64]                                                | Output this tensor when cacheMode≠0. Data type consistent with input.                               |
|                 | outTensor3 | kv_cache_out1  | float16/bf16       | ND/NZ      | cache_mode=1：<br>[blockNum,blockSize,1,64]<br>cache_mode=2/3：<br>[blockNum, headNum*64/16, block_size, 16] | Output this tensor when cacheMode≠0. Data type consistent with input. When cache_mode=2, data format is NZ. When cache_mode=3, format is NZ. Same tensor as input kvCacheRope. |

## Specification Constraints
1. tokenNum <= 1024
2. blockSize <= 128 or = 256
3. When cache_mode=2 or 3, blockSize = 128

## Hardware Support Status
| Hardware Model       | Support Status |
|----------------------|----------------|
| Atlas 800 A2/A3      | Supported      |
| Atlas Inference Series| Not Supported  |
