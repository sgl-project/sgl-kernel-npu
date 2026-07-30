# Fused Deep MoE API

`Buffer.fused_deep_moe(...)` is the unified fused MoE entrypoint in DeepEP-Ascend.
It now supports two execution backends:

- `deep_ep`: legacy fused kernels exposed by `deep_ep_cpp`
- `mega_moe`: `cann_ops_transformer.ops.mega_moe`

`backend="auto"` keeps the existing A5 behavior and routes non-A5 or mega_moe-only features to `mega_moe`.

## Python API

```python
def fused_deep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gmm1_permuted_weight: Union[torch.Tensor, List[torch.Tensor]],
    gmm1_permuted_weight_scale: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    gmm2_weight: Union[torch.Tensor, List[torch.Tensor]],
    gmm2_weight_scale: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    num_max_dispatch_tokens_per_rank: int,
    num_experts: int,
    quant_mode: int = 1,
    fuse_mode: FuseMode = FuseMode.FUSED_DEEP_MOE,
    backend: str = "auto",
    activation: str = "swiglu",
    linear_beta: Optional[float] = None,
    l1_bias: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    l2_bias: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    dispatch_quant_mode: Optional[int] = None,
    dispatch_quant_out_dtype: Optional[torch.dtype] = None,
    max_recv_token_num: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]
```

## Backend Rules

### `backend="auto"`

- A5 + legacy-compatible arguments: use `deep_ep`
- `activation="situ"`: use `mega_moe`
- `l1_bias` or `l2_bias` provided: use `mega_moe`
- `dispatch_quant_mode` explicitly provided: use `mega_moe`
- non-A5 build: use `mega_moe`

### `backend="deep_ep"`

- Supports `FuseMode.FUSED_DEEP_MOE`
- Supports `FuseMode.DISPATCH_FFN_COMBINE`
- Supports only `activation="swiglu"`
- Uses legacy `quant_mode`
- Requires tensor-form legacy fused weights/scales
- Does not support `linear_beta`, `l1_bias`, `l2_bias`, `dispatch_quant_mode`,
  `dispatch_quant_out_dtype`, or `max_recv_token_num`

### `backend="mega_moe"`

- Supports only `FuseMode.FUSED_DEEP_MOE`
- Requires `cann_ops_transformer`
- Supports `activation="swiglu"`, `activation="swiglu_gpt_oss"`, and `activation="situ"`
- Interprets legacy parameter names as mega_moe inputs:
  - `gmm1_permuted_weight -> l1_weights`
  - `gmm1_permuted_weight_scale -> l1_weights_sf`
  - `gmm2_weight -> l2_weights`
  - `gmm2_weight_scale -> l2_weights_sf`
- Accepts either:
  - a tensor whose leading dimension is the local expert count
  - or `list[Tensor]` with one tensor per local expert

## Parameter Notes

| Parameter | Notes |
|---|---|
| `x` | `[bs, hidden]` token tensor. |
| `topk_idx` | `[bs, num_topk]` routing indices. `-1` is allowed. |
| `topk_weights` | `[bs, num_topk]` combine weights. |
| `gmm1_permuted_weight` | Legacy name kept for compatibility. On `mega_moe`, it must contain first linear weights in layout `[hidden, 2 * intermediate_hidden]` per expert. |
| `gmm1_permuted_weight_scale` | Optional on `mega_moe` A16W16. Required on `mega_moe` A8W8-INT/A8W4-INT. |
| `gmm2_weight` | Legacy name kept for compatibility. On `mega_moe`, it must contain second linear weights in layout `[intermediate_hidden, hidden]` per expert. |
| `gmm2_weight_scale` | Optional on `mega_moe` A16W16. Required on `mega_moe` A8W8-INT/A8W4-INT. |
| `num_max_dispatch_tokens_per_rank` | EP dispatch capacity hint shared across ranks. |
| `num_experts` | Global expert count. On `mega_moe`, it must be divisible by the process-group size. |
| `quant_mode` | Legacy deep_ep-only quantization selector. |
| `fuse_mode` | `mega_moe` supports only `FuseMode.FUSED_DEEP_MOE`. |
| `backend` | `"auto"`, `"deep_ep"`, or `"mega_moe"`. |
| `activation` | `mega_moe` supports `"swiglu"`, `"swiglu_gpt_oss"`, `"situ"`. `deep_ep` supports only `"swiglu"`. |
| `linear_beta` | Public replacement for the old `activation_clamp` naming. Only meaningful for `activation="situ"` on `mega_moe`. Internally forwarded as `activation_clamp=linear_beta` when calling mega_moe. |
| `l1_bias`, `l2_bias` | Optional A8W4-INT compensation biases for `mega_moe` only. |
| `dispatch_quant_mode` | `mega_moe` wrapper currently supports `0` (A16W16) and `2` (A8W8-INT/A8W4-INT). |
| `dispatch_quant_out_dtype` | `mega_moe` wrapper currently supports only `torch.int8` when `dispatch_quant_mode=2`. |
| `max_recv_token_num` | Forwarded to mega_moe SymmBuffer creation only. |

## Mega MoE Quantized Scenes

### A16W16

- `dispatch_quant_mode=0`
- `dispatch_quant_out_dtype=None`
- `gmm1_permuted_weight_scale=None`
- `gmm2_weight_scale=None`
- `l1_bias=None`
- `l2_bias=None`

### A8W8-INT

- `dispatch_quant_mode=2`
- `dispatch_quant_out_dtype=torch.int8`
- `gmm1_permuted_weight_scale` required
- `gmm2_weight_scale` required
- `l1_bias=None`
- `l2_bias=None`

### A8W4-INT

- `dispatch_quant_mode=2`
- `dispatch_quant_out_dtype=torch.int8`
- `gmm1_permuted_weight_scale` required
- `gmm2_weight_scale` required
- `l1_bias` required
- `l2_bias` required

## `situ` Activation

`situ` is not implemented in `buffer.py` itself. It is forwarded to `mega_moe`.

- Public API name: `linear_beta`
- mega_moe call name: `activation_clamp`
- `linear_beta=None` or `0`: no extra linear beta term
- `linear_beta>0`: forwarded as the `situ` linear beta control value

## Return Value

`fused_deep_moe(...)` returns `(output, aux)`:

- `output`: fused MoE output with shape `[bs, hidden]`
- `aux`:
  - `deep_ep + FuseMode.FUSED_DEEP_MOE`: `ep_recv_count`
  - `deep_ep + FuseMode.DISPATCH_FFN_COMBINE`: `expert_token_nums`
  - `mega_moe`: `expert_token_nums`

## Dependency Note

The `mega_moe` backend depends on:

```python
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
```

If that package is not available, `deep_ep` backend behavior is unchanged, but calls that
actually route to `mega_moe` will raise an import error with an explicit message.
