## 功能
融合了MLA场景下PagedAttention输入数据处理的全过程，包括从隐状态输入开始经过rmsnorm、反量化、matmul、rope、reshapeAndCache的一系列计算。

### 图1 计算流程图
![pic](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/83RC1alpha001/API/ascendtbapi/figure/zh-cn_image_0000002374942716.png "")

## 输入输出列表
### 输入Tensor
| 所属模块功能       | 标识        | 名称          | 数据类型             | 数据格式 | 维度                                      | 详细描述                                                                 |
|--------------------|-------------|---------------|----------------------|----------|-------------------------------------------|--------------------------------------------------------------------------|
| 输入数据            | inTensor0   | hiddenState    | float16/bf16         | ND       | [tokenNum, hiddenSize]，hiddenSize取值：[2048,8192] | 必选。                                                                   |
| rmsNormQuant_0     | inTensor1   | gamma0        | float16/bf16         | ND       | [hiddenSize]，hiddenSize取值：[2048,8192]  | 必选。数据类型与input一致。                                              |
|                    | inTensor2   | beta0         | float16/bf16         | ND       | [hiddenSize]，hiddenSize取值：[2048,8192]  | 必选。数据类型与input一致。                                              |
|                    | inTensor3   | quant_scale0  | float16/bf16         | ND       | [1]                                       | 必选，支持传入空tensor。仅在quantMode为0时传入，数据类型与input一致。     |
|                    | inTensor4   | quant_offset0 | int8                 | ND       | [1]                                       | 必选，支持传入空tensor。仅在quantMode为0时传入                           |
| matmul_0           | inTensor5   | wdqkv         | int8<br>float16/bf16 | NZ       | [1,224,2112,32]                           | 必选。<br>数据类型与input一致时不开启rmsNormQuant。                       |
|                    | inTensor6   | descale0      | int64/float          | ND       | [2112]                                    | 必选。input为fp16时为int64，input为bf16时为float                         |
|                    | inTensor7   | bias0         | int32                | ND       | [2112]                                    | 必选，支持传入空tensor。quantMode为1、3时不传入                          |
| rmsNormQuant_1     | inTensor8   | gamma1        | float16/bf16         | ND       | [1536]                                    | 必选。数据类型与input一致。                                              |
|                    | inTensor9   | beta1         | float16/bf16         | ND       | [1536]                                    | 必选。数据类型与input一致。                                              |
|                    | inTensor10  | quant_scale1  | float16/bf16         | ND       | [1]                                       | 必选，支持传入空tensor。仅在quantMode为0时传入，数据类型与input一致。     |
|                    | inTensor11  | quant_offset1 | int8                 | ND       | [1]                                       | 必选，支持传入空tensor。仅在quantMode为0时传入                           |
| matmul_1           | inTensor12  | wuq           | int8<br>float16/bf16 | NZ       | [1,48,headNum*192,32]                     | 必选。<br>数据类型与input一致时不开启rmsNormQuant。                       |
|                    | inTensor13  | descale1      | int64/float          | ND       | [headNum*192]                             | 必选。input为fp16时为int64，input为bf16时为float                         |
|                    | inTensor14  | bias1         | int32                | ND       | [headNum*192]                             | 必选，支持传入空tensor。quantMode为1、3时不传入                          |
| rmsNorm            | inTensor15  | gamma2        | float16/bf16         | ND       | [512]                                     | 必选。数据类型与input一致。                                              |
| rope               | inTensor16  | cos           | float16/bf16         | ND       | [tokenNum,64]                             | 必选。数据类型与input一致。                                              |
|                    | inTensor17  | sin           | float16/bf16         | ND       | [tokenNum,64]                             | 必选。数据类型与input一致。                                              |
| matmulEin          | inTensor18  | wuk           | float16/bf16         | ND/NZ    | ND:[headNum,128,512]<br>NZ:[headNum,32,128,16] | 必选。数据类型与input一致。                                              |
| reshapeAndCache    | inTensor19  | kv_cache      | float16/bf16/int8    | ND/NZ    | cacheMode为0：<br>[blockNum,blockSize,1,576]<br>cacheMode为1：<br>[blockNum,blockSize,1,512]<br>cacheMode为2：<br>[blockNum, 1*512/32, block_size, 32]<br>cacheMode为3：<br>[blockNum, 1*512/16, block_size, 16] | 必选。数据类型与input一致。<br>cacheMode为1时，tensor的shape为拆分情况。cacheMode为2时格式为NZ，类型为int8。cacheMode为3时，格式为NZ。 |
|                    | inTensor20  | kv_cache_rope | float16/bf16         | ND/NZ    | cacheMode为1：<br>[blockNum,blockSize,1,64]<br>cacheMode为2或3：<br>[blockNum, headNum*64 / 16 ,block_size, 16] | 必选，支持传入空tensor。cacheMode不为0时传入，数据类型与input一致。cacheMode为2或3时，格式为NZ。 |
|                    | inTensor21  | slotmapping   | int32                | ND       | [tokenNum]                                | 必选。                                                                   |
| quant              | inTensor22  | ctkv_scale    | float16/bf16         | ND       | [1]                                       | 必选，支持传入空tensor。cacheMode为2时传入，数据类型与input一致。         |
|                    | inTensor23  | q_nope_scale  | float16/bf16         | ND       | [headNum]                                 | 必选，支持传入空tensor。cacheMode为2时传入，数据类型与input一致。         |

### cache_mode
cache模式
cache_mode为1时，输入输出的kcache拆分为krope和ctkv，q拆分为qrope和qnope。\
cache_mode为2时，cache_mode为1的基础上，krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型。\
cache_mode为3时，cache_mode为1的基础上，krope和ctkv转为NZ格式输出。
### quant_mode
RmsNorm量化类型
PER_TENSOR_QUANT_ASYMM，per_tensor静态非对称量化，默认量化类型；\
PER_TOKEN_QUANT_SYMM，per_token动态对称量化；


## 输出Tensor
可选输出tensor不能使用空tensor占位
| 所属模块功能 | 标识       | 名称          | 数据类型           | 数据格式   | 维度                                                                 | 详细描述                                                                                   |
|--------------|------------|---------------|--------------------|----------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 输出数据     | outTensor0 | q_out0         | float16/bf16/int8  | ND       | cacheMode为0：<br>[tokenNum,headNum,576]<br>cacheMode为1或2或3：<br>[tokenNum,headNum,512] | 输出tensor。数据类型与input一致。cacheMode为2时数据类型为int8。                             |
|              | outTensor1 | kv_cache_out0 | float16/bf16/int8  | ND/NZ    | cacheMode为0：<br>[blockNum,blockSize,1,576]<br>cacheMode为1：<br>[blockNum,blockSize,1,512]<br>cacheMode为2：<br>[blockNum, headNum*512/32,block_size, 32]<br>cacheMode为3：<br>[blockNum, headNum*512/16,block_size, 16] | 输出tensor。数据类型与input一致。cacheMode为2时数据类型为int8，格式为NZ。cacheMode为3时，格式为NZ。与输入的kvCache为同一tensor。 |
|              | outTensor2 | q_out1        | float16/bf16       | ND       | [tokenNum,headNum,64]                                                | cacheMode不为0时输出此tensor。数据类型与input一致。                                        |
|              | outTensor3 | kv_cache_out1 | float16/bf16       | ND/NZ    | cacheMode为1：<br>[blockNum,blockSize,1,64]<br>cacheMode为2或3：<br>[blockNum, headNum*64 / 16 ,block_size, 16] | cacheMode不为0时输出此tensor。数据类型与input一致。cacheMode为2时数据格式为NZ。cacheMode为3时，格式为NZ。与输入kvCacheRope为同一tensor。 |

## 规格约束
1、tokenNum <= 1024 \
2、blockSize <= 128 或 = 256 \
3、cacheMode为2或3时，blockSize = 128

## 硬件支持情况
| 硬件型号         | 支持情况 |
|-----------------|----------|
|Atlas 800 A2/A3  | 支持     |
|Atlas 推理系列产品| 不支持   |
