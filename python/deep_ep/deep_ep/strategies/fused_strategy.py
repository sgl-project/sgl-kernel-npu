"""
Fused MoE execution strategies.
All fused mode strategy implementations are in this file.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from ..ep_strategy import FusedEPCommStrategy, register_fused_strategy

TensorOrTensors = Union[torch.Tensor, List[torch.Tensor]]
_MEGA_MOE_MXFP_MODES = frozenset(
    {"mx_fp8_e4m3", "mx_fp8_e5m2", "mx_fp4_e2m1", "mx_fp8_e4m3_w4"}
)


@register_fused_strategy("fused_deep_moe")
class FusedDeepMoeStrategy(FusedEPCommStrategy):
    """Fused MoE strategy backed by aclnnFusedDeepMoe."""

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime

    def get_name(self) -> str:
        return "fused_deep_moe"

    def get_supported_modes(self) -> List[str]:
        return ["fused"]

    def fused_moe(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight: TensorOrTensors,
        gmm1_permuted_weight_scale: Optional[TensorOrTensors],
        gmm2_weight: TensorOrTensors,
        gmm2_weight_scale: Optional[TensorOrTensors],
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        quant_mode: int,
        activation: Optional[str],
        beta: Optional[float],
        linear_beta: Optional[float],
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
        profile_enable: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, ep_recv_count = self.runtime.fused_deep_moe(
            x,
            topk_idx.int(),
            gmm1_permuted_weight,
            gmm1_permuted_weight_scale,
            gmm2_weight,
            gmm2_weight_scale,
            topk_weights,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            quant_mode,
            profile_enable,
            activation,
            beta,
            linear_beta,
        )
        return output, ep_recv_count


@register_fused_strategy("dispatch_ffn_combine")
class DispatchFFNCombineStrategy(FusedEPCommStrategy):
    """Fused MoE strategy backed by aclnnDispatchFFNCombine."""

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self.runtime = runtime

    def get_name(self) -> str:
        return "dispatch_ffn_combine"

    def get_supported_modes(self) -> List[str]:
        return ["fused"]

    def fused_moe(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight: TensorOrTensors,
        gmm1_permuted_weight_scale: Optional[TensorOrTensors],
        gmm2_weight: TensorOrTensors,
        gmm2_weight_scale: Optional[TensorOrTensors],
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        quant_mode: int,
        activation: Optional[str],
        beta: Optional[float],
        linear_beta: Optional[float],
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
        profile_enable: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if activation == "situ":
            raise NotImplementedError(
                "SiTU is only supported by FuseMode.FUSED_DEEP_MOE and "
                "FuseMode.MEGA_MOE"
            )
        output, expert_token_nums = self.runtime.dispatch_ffn_combine(
            x,
            topk_idx.int(),
            gmm1_permuted_weight,
            gmm1_permuted_weight_scale,
            gmm2_weight,
            gmm2_weight_scale,
            topk_weights,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            quant_mode,
        )
        return output, expert_token_nums


@register_fused_strategy("mega_moe")
class MegaMoeStrategy(FusedEPCommStrategy):
    """Fused MoE strategy backed by cann_ops_transformer.ops.mega_moe."""

    def __init__(self, runtime, group: dist.ProcessGroup):
        super().__init__(group)
        self._mega_moe_symm_buffers: Dict[Tuple, object] = {}

    def get_name(self) -> str:
        return "mega_moe"

    def get_supported_modes(self) -> List[str]:
        return ["fused"]

    def destroy(self) -> None:
        for symm_buffer in self._mega_moe_symm_buffers.values():
            try:
                symm_buffer.destroy()
            except Exception:
                pass
        self._mega_moe_symm_buffers.clear()

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass

    @staticmethod
    def _load_mega_moe_ops():
        # mega_moe is JIT-built (requires ninja), so keep the import lazy. This
        # lets callers use the other fused modes without the optional dependency.
        try:
            from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
        except (ImportError, RuntimeError) as e:
            raise RuntimeError(
                "Failed to import `cann_ops_transformer.ops.mega_moe`, which "
                "is required by FuseMode.MEGA_MOE. The mega_moe op is JIT-built "
                "with ninja; ensure `ninja` is installed "
                "(e.g. `pip install ninja`). Original error: "
                f"{e}"
            ) from e
        return get_symm_buffer_for_mega_moe, mega_moe

    @staticmethod
    def _fp4_type() -> int:
        import torch_npu

        dtype = getattr(torch_npu, "float4_e2m1fn_x2", None)
        if type(dtype) is not int:
            raise RuntimeError("MXFP4 MegaMoe requires torch_npu.float4_e2m1fn_x2")
        return dtype

    def _resolve_mega_moe_quant(self, quant_mode: Union[int, str]):
        if type(quant_mode) is int and quant_mode in (0, 1):
            mode = "bf16" if quant_mode == 0 else "int8"
        elif isinstance(quant_mode, str) and quant_mode in (
            "bf16",
            "int8",
            *_MEGA_MOE_MXFP_MODES,
        ):
            mode = quant_mode
        else:
            raise ValueError(f"Unsupported quant_mode for MEGA_MOE: {quant_mode!r}")

        if mode == "bf16":
            return 0, None, None
        if mode == "int8":
            return 2, torch.int8, None
        if mode == "mx_fp4_e2m1":
            # CANN 9.2.0 accepts the torch_npu integer enum, not a torch dtype.
            fp4_type = self._fp4_type()
            return 4, fp4_type, fp4_type

        name = (
            "float8_e4m3fn"
            if mode in ("mx_fp8_e4m3", "mx_fp8_e4m3_w4")
            else "float8_e5m2"
        )
        dtype = getattr(torch, name, None)
        if dtype is None:
            raise RuntimeError(f"MXFP MegaMoe requires torch.{name}")
        weight_type = self._fp4_type() if mode == "mx_fp8_e4m3_w4" else None
        return 4, dtype, weight_type

    def fused_moe(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight: TensorOrTensors,
        gmm1_permuted_weight_scale: Optional[TensorOrTensors],
        gmm2_weight: TensorOrTensors,
        gmm2_weight_scale: Optional[TensorOrTensors],
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        quant_mode: Union[int, str],
        activation: Optional[str],
        beta: Optional[float],
        linear_beta: Optional[float],
        l1_bias: Optional[TensorOrTensors],
        l2_bias: Optional[TensorOrTensors],
        profile_enable: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dispatch_quant_mode, dispatch_quant_out_dtype, fp4_weight_type = (
            self._resolve_mega_moe_quant(quant_mode)
        )
        is_mxfp = dispatch_quant_mode == 4
        is_fp4_weight = fp4_weight_type is not None
        if not isinstance(gmm2_weight, list) or len(gmm2_weight) == 0:
            raise ValueError(
                "FuseMode.MEGA_MOE requires gmm2_weight to be a non-empty "
                "list of per-expert tensors"
            )
        weight = gmm2_weight[0]
        if is_mxfp:
            pack_factor = 2 if is_fp4_weight else 1
            intermediate_hidden = weight.shape[-1] * pack_factor
        else:
            intermediate_hidden = weight.shape[-2]
        hidden = x.size(1)
        num_tokens = x.size(0)
        if num_tokens > num_max_dispatch_tokens_per_rank:
            raise ValueError(
                "The number of input tokens exceeds "
                f"`num_max_dispatch_tokens_per_rank`: "
                f"{num_tokens} > {num_max_dispatch_tokens_per_rank}."
            )
        # CANN validates device support and platform-specific parameter constraints.
        get_symm_buffer_for_mega_moe, mega_moe = self._load_mega_moe_ops()
        cache_key = (
            num_experts,
            num_max_dispatch_tokens_per_rank,
            topk_idx.size(1),
            hidden,
            intermediate_hidden,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
        )
        symm_buffer = self._mega_moe_symm_buffers.get(cache_key)
        if symm_buffer is None:
            symm_buffer = get_symm_buffer_for_mega_moe(
                self.group,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                num_topk=topk_idx.size(1),
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                max_recv_token_num=0,
                dispatch_quant_mode=dispatch_quant_mode,
                dispatch_quant_out_dtype=dispatch_quant_out_dtype,
            )
            self._mega_moe_symm_buffers[cache_key] = symm_buffer

        topk_ids = topk_idx.int()
        x_active_mask = None
        if not is_mxfp:
            x_active_mask = torch.zeros(
                num_max_dispatch_tokens_per_rank,
                dtype=torch.int8,
                device=x.device,
            )
            x_active_mask[:num_tokens] = 1
        if not is_mxfp and num_tokens < num_max_dispatch_tokens_per_rank:
            padding_size = num_max_dispatch_tokens_per_rank - num_tokens
            x = torch.cat(
                (x, x.new_zeros((padding_size, hidden))),
                dim=0,
            )
            topk_ids = torch.cat(
                (
                    topk_ids,
                    topk_ids.new_zeros((padding_size, topk_ids.size(1))),
                ),
                dim=0,
            )
            topk_weights = torch.cat(
                (
                    topk_weights,
                    topk_weights.new_zeros((padding_size, topk_weights.size(1))),
                ),
                dim=0,
            )

        activation_params = None
        if activation == "situ":
            activation_params = {}
            if beta is not None:
                activation_params["beta"] = beta
            if linear_beta is not None:
                activation_params["linear_beta"] = linear_beta
            activation_params = activation_params or None

        weight_type_kwargs = {}
        if is_mxfp:
            weight_type_kwargs = {
                "weight1_type": fp4_weight_type,
                "weight2_type": fp4_weight_type,
            }
        output, expert_token_nums = mega_moe(
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            l1_weights=gmm1_permuted_weight,
            l2_weights=gmm2_weight,
            sym_buffer=symm_buffer,
            l1_weights_sf=gmm1_permuted_weight_scale,
            l2_weights_sf=gmm2_weight_scale,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            x_active_mask=x_active_mask,
            activation={
                "situ": "situglu",
                "swiglu": "swiglu",
            }.get(activation, activation),
            activation_params=activation_params,
            **weight_type_kwargs,
        )
        return (output if is_mxfp else output[:num_tokens]), expert_token_nums
