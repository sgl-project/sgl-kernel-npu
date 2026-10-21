# RMSNorm+bias sample tests

No direct use of this standalone `rmsnorm_bias` was found in SGLang revision
`86624f14eb`. ModelSlim has a mathematically similar
[`npu_rms_norm(...) + bias` path](https://github.com/sgl-project/sglang/blob/86624f14eb/python/sglang/srt/layers/quantization/modelslim/modelslim.py#L60).
These are sample interface tests, not a reproduction of a deployed model path.

## Shapes

| Shape | Source / purpose |
| --- | --- |
| `(1, 4096)` | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) hidden size; representative single-token decode. |
| `(16, 7168)` | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) hidden size; representative batched decode. |
| `(1, 1)` | Minimum nonempty reduction; short-input boundary. |
| `(7, 1025)` | Odd row count, reduction padding, and a one-element tail beyond the 1024-element quantization tile. |

Only the hidden widths come from model configs; row counts are sample workloads.
Bias and int8 quantization coverage does not imply these checkpoints use this
interface. Empty inputs and noncontiguous layouts are outside this sample's scope.

## Modes

| Mode | Reason |
| --- | --- |
| `normal` | Random inputs, weights and bias exercise general affine RMSNorm. |
| `zero` | Zero variance exercises eps handling, finite output and bias preservation. |
| `offset_only` | Offset without scale must be ignored and must not enable int8 output. |
| `quantized` | Signed unit inputs and controlled scale/offset exercise truncation, int8 saturation and tile masks. Avoids ambiguous rounding boundaries; width 1 covers only the positive side. |

Four shapes × four modes × three dtypes × two eps values = 96 cases.
Accuracy compares `auto` with `torch_native` by default. Performance collection
defaults to shapes in `perf_shape` with FP16 and `quantized` mode;
`--enable-perf=true/false` overrides this selection for all cases.
