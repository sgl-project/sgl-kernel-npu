"""
Fused MoE execution strategies.
"""

from typing import Dict, List, Optional, Tuple

import torch

from ..ep_strategy import FusedEPStrategy, register_fused_strategy

try:
    from cann_ops_transformer.ops import (
        get_symm_buffer_for_mega_moe as _get_symm_buffer_for_mega_moe,
    )
    from cann_ops_transformer.ops import mega_moe as _mega_moe

    _MEGA_MOE_IMPORT_ERROR = None
except ImportError as exc:
    _get_symm_buffer_for_mega_moe = None
    _mega_moe = None
    _MEGA_MOE_IMPORT_ERROR = exc


@register_fused_strategy("deep_ep")
class DeepEPFusedStrategy(FusedEPStrategy):
    """Fused MoE strategy backed by deep_ep runtime kernels."""

    def get_name(self) -> str:
        return "deep_ep"

    def run(
        self,
        *,
        buffer,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        quant_mode: int,
        fuse_mode,
        activation: str,
        activation_clamp: Optional[float],
        beta: float,
        linear_beta: Optional[float],
        l1_bias,
        l2_bias,
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
        max_recv_token_num: int,
    ):
        activation_clamp = buffer._validate_activation_clamp(activation_clamp)
        if not buffer._is_default_beta(beta):
            raise ValueError("`beta` is only supported by the mega_moe backend.")
        if activation != "situ" and not buffer._is_zero_like_linear_beta(linear_beta):
            raise ValueError('`linear_beta` is only valid when `activation="situ"`.')
        if activation != "swiglu":
            raise ValueError(
                "The deep_ep fused backend only supports activation='swiglu'. "
                "Use backend='mega_moe' for other activation types."
            )
        if activation_clamp is not None:
            raise ValueError(
                "`activation_clamp` is only supported by the mega_moe backend."
            )
        if not buffer._is_zero_like_linear_beta(linear_beta):
            raise ValueError("`linear_beta` is only supported by the mega_moe backend.")
        if l1_bias is not None or l2_bias is not None:
            raise ValueError(
                "`l1_bias` and `l2_bias` are only supported by the mega_moe backend."
            )
        if dispatch_quant_mode is not None or dispatch_quant_out_dtype is not None:
            raise ValueError(
                "`dispatch_quant_mode` and `dispatch_quant_out_dtype` are only "
                "supported by the mega_moe backend."
            )
        if isinstance(gmm1_permuted_weight, list) or isinstance(gmm2_weight, list):
            raise TypeError(
                "The deep_ep fused backend expects Tensor inputs for "
                "`gmm1_permuted_weight` and `gmm2_weight`."
            )
        if isinstance(gmm1_permuted_weight_scale, list) or isinstance(
            gmm2_weight_scale, list
        ):
            raise TypeError(
                "The deep_ep fused backend expects Tensor inputs for weight scales."
            )
        if gmm1_permuted_weight_scale is None or gmm2_weight_scale is None:
            raise ValueError(
                "The deep_ep fused backend requires both weight scale tensors."
            )

        topk_ids = topk_idx.int()
        if fuse_mode == buffer.FuseMode.FUSED_DEEP_MOE:
            gmm1_permuted_weight_scale = gmm1_permuted_weight_scale.float()
            gmm2_weight_scale = gmm2_weight_scale.float()
            output, ep_recv_count = buffer.runtime.fused_deep_moe(
                x,
                topk_ids,
                gmm1_permuted_weight,
                gmm1_permuted_weight_scale,
                gmm2_weight,
                gmm2_weight_scale,
                topk_weights,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                quant_mode,
            )
            return output, ep_recv_count
        if fuse_mode == buffer.FuseMode.DISPATCH_FFN_COMBINE:
            max_output_size = num_max_dispatch_tokens_per_rank
            output, expert_token_nums = buffer.runtime.dispatch_ffn_combine(
                x,
                topk_ids,
                gmm1_permuted_weight,
                gmm1_permuted_weight_scale,
                gmm2_weight,
                gmm2_weight_scale,
                topk_weights,
                max_output_size,
                num_experts,
                quant_mode,
            )
            return output, expert_token_nums
        raise NotImplementedError(f"Not support fuse_mode:{fuse_mode}")


@register_fused_strategy("mega_moe")
class MegaMoeFusedStrategy(FusedEPStrategy):
    """Fused MoE strategy backed by cann_ops_transformer mega_moe."""

    def __init__(self) -> None:
        self._symm_buffer_cache: Dict[tuple, object] = {}

    def get_name(self) -> str:
        return "mega_moe"

    def destroy(self) -> None:
        for sym_buffer in self._symm_buffer_cache.values():
            try:
                sym_buffer.destroy()
            except Exception:
                pass
        self._symm_buffer_cache.clear()

    @staticmethod
    def _require_mega_moe_ops():
        if _get_symm_buffer_for_mega_moe is None or _mega_moe is None:
            raise ImportError(
                "The mega_moe backend requires the optional dependency "
                "`cann_ops_transformer`. Install or expose `cann_ops_transformer.ops` "
                'before calling `Buffer.fused_deep_moe(..., backend="mega_moe")`.'
            ) from _MEGA_MOE_IMPORT_ERROR
        return _get_symm_buffer_for_mega_moe, _mega_moe

    @staticmethod
    def _normalize_expert_param(
        param,
        name: str,
        expected_num_local_experts: int,
    ) -> Optional[List[torch.Tensor]]:
        if param is None:
            return None
        if isinstance(param, list):
            if len(param) != expected_num_local_experts:
                raise ValueError(
                    f"`{name}` must contain exactly {expected_num_local_experts} "
                    f"local expert tensors, but got {len(param)}."
                )
            return param
        if not isinstance(param, torch.Tensor):
            raise TypeError(
                f"`{name}` must be a Tensor, a list[Tensor], or None, "
                f"but got {type(param)}."
            )
        if param.dim() == 0:
            raise ValueError(f"`{name}` must not be a scalar tensor.")
        if param.size(0) != expected_num_local_experts:
            raise ValueError(
                f"`{name}` must have leading dimension {expected_num_local_experts} "
                f"for local experts, but got shape {tuple(param.shape)}."
            )
        return [param[i] for i in range(expected_num_local_experts)]

    @staticmethod
    def _infer_mega_moe_quant_config(
        l1_weights_sf,
        l2_weights_sf,
        l1_bias,
        l2_bias,
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
    ):
        has_scales = l1_weights_sf is not None or l2_weights_sf is not None
        has_bias = l1_bias is not None or l2_bias is not None
        if has_scales and (l1_weights_sf is None or l2_weights_sf is None):
            raise ValueError(
                "`gmm1_permuted_weight_scale` and `gmm2_weight_scale` must both be "
                "provided for mega_moe quantized execution."
            )
        if has_bias and (l1_bias is None or l2_bias is None):
            raise ValueError(
                "`l1_bias` and `l2_bias` must both be provided for A8W4-INT "
                "mega_moe execution."
            )

        resolved_quant_mode = (
            2
            if has_scales or has_bias
            else 0 if dispatch_quant_mode is None else dispatch_quant_mode
        )
        if resolved_quant_mode not in (0, 2):
            raise ValueError(
                "`dispatch_quant_mode` only supports 0 (A16W16) or 2 "
                "(A8W8-INT/A8W4-INT) in fused_deep_moe mega_moe backend."
            )
        if resolved_quant_mode == 0:
            if has_scales or has_bias:
                raise ValueError(
                    "Scale and bias tensors are only valid when "
                    "`dispatch_quant_mode=2`."
                )
            if dispatch_quant_out_dtype is not None:
                raise ValueError(
                    "`dispatch_quant_out_dtype` must be None when "
                    "`dispatch_quant_mode=0`."
                )
            return 0, None

        resolved_quant_out_dtype = (
            torch.int8 if dispatch_quant_out_dtype is None else dispatch_quant_out_dtype
        )
        if resolved_quant_out_dtype != torch.int8:
            raise ValueError(
                "`dispatch_quant_out_dtype` must be torch.int8 for "
                "A8W8-INT/A8W4-INT mega_moe execution."
            )
        return resolved_quant_mode, resolved_quant_out_dtype

    def _prepare_quant_scene(
        self,
        *,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        l1_bias,
        l2_bias,
        expected_num_local_experts: int,
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
    ):
        l1_weights = self._normalize_expert_param(
            gmm1_permuted_weight,
            "gmm1_permuted_weight",
            expected_num_local_experts,
        )
        l2_weights = self._normalize_expert_param(
            gmm2_weight,
            "gmm2_weight",
            expected_num_local_experts,
        )
        l1_weights_sf = self._normalize_expert_param(
            gmm1_permuted_weight_scale,
            "gmm1_permuted_weight_scale",
            expected_num_local_experts,
        )
        l2_weights_sf = self._normalize_expert_param(
            gmm2_weight_scale,
            "gmm2_weight_scale",
            expected_num_local_experts,
        )
        l1_bias_list = self._normalize_expert_param(
            l1_bias,
            "l1_bias",
            expected_num_local_experts,
        )
        l2_bias_list = self._normalize_expert_param(
            l2_bias,
            "l2_bias",
            expected_num_local_experts,
        )
        resolved_dispatch_quant_mode, resolved_dispatch_quant_out_dtype = (
            self._infer_mega_moe_quant_config(
                l1_weights_sf,
                l2_weights_sf,
                l1_bias_list,
                l2_bias_list,
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
            )
        )
        return (
            l1_weights,
            l2_weights,
            l1_weights_sf,
            l2_weights_sf,
            l1_bias_list,
            l2_bias_list,
            resolved_dispatch_quant_mode,
            resolved_dispatch_quant_out_dtype,
        )

    def _get_or_create_mega_moe_symm_buffer(
        self,
        *,
        group,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        max_recv_token_num: int,
        dispatch_quant_mode: int,
        dispatch_quant_out_dtype: Optional[torch.dtype],
        activation: str,
    ):
        get_symm_buffer_for_mega_moe, _ = self._require_mega_moe_ops()
        cache_key = (
            num_experts,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            max_recv_token_num,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
            activation,
        )
        sym_buffer = self._symm_buffer_cache.get(cache_key)
        if sym_buffer is None:
            sym_buffer = get_symm_buffer_for_mega_moe(
                group,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                max_recv_token_num=max_recv_token_num,
                dispatch_quant_mode=dispatch_quant_mode,
                dispatch_quant_out_dtype=dispatch_quant_out_dtype,
            )
            self._symm_buffer_cache[cache_key] = sym_buffer
        return sym_buffer

    def _pad_mega_moe_inputs(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        original_num_tokens = x.size(0)
        if original_num_tokens > num_max_dispatch_tokens_per_rank:
            raise ValueError(
                "The number of input tokens exceeds "
                "`num_max_dispatch_tokens_per_rank`: "
                f"{original_num_tokens} > {num_max_dispatch_tokens_per_rank}."
            )

        x_active_mask = torch.zeros(
            num_max_dispatch_tokens_per_rank,
            dtype=torch.int8,
            device=x.device,
        )
        x_active_mask[:original_num_tokens] = 1

        if original_num_tokens == num_max_dispatch_tokens_per_rank:
            return x, topk_idx, topk_weights, x_active_mask, original_num_tokens

        padding_size = num_max_dispatch_tokens_per_rank - original_num_tokens
        x_padded = torch.cat((x, x.new_zeros((padding_size, x.size(1)))), dim=0)
        topk_idx_padded = torch.cat(
            (topk_idx, topk_idx.new_zeros((padding_size, topk_idx.size(1)))),
            dim=0,
        )
        topk_weights_padded = torch.cat(
            (
                topk_weights,
                topk_weights.new_zeros((padding_size, topk_weights.size(1))),
            ),
            dim=0,
        )
        return (
            x_padded,
            topk_idx_padded,
            topk_weights_padded,
            x_active_mask,
            original_num_tokens,
        )

    def _validate_weight_layout(
        self,
        *,
        l1_weights,
        l2_weights,
        hidden: int,
        resolved_dispatch_quant_mode: int,
        l1_bias_list,
        l2_bias_list,
    ):
        is_a8w4_int = l1_bias_list is not None and l2_bias_list is not None
        if not l2_weights or l2_weights[0].dim() < 2:
            raise ValueError(
                "`gmm2_weight` must contain per-expert 2D tensors in mega_moe layout "
                "[intermediate_hidden, hidden]."
            )
        intermediate_hidden = l2_weights[0].shape[-2]
        expected_l2_last_dim = hidden // 8 if is_a8w4_int else hidden
        expected_l1_last_dim = (
            (intermediate_hidden * 2) // 8 if is_a8w4_int else intermediate_hidden * 2
        )
        inferred_scene = (
            "A8W4-INT"
            if is_a8w4_int
            else "A8W8-INT" if resolved_dispatch_quant_mode == 2 else "A16W16"
        )
        if l2_weights[0].shape[-1] != expected_l2_last_dim:
            raise ValueError(
                "`gmm2_weight` has an invalid mega_moe layout for "
                f"{inferred_scene}. Expected first local expert shape "
                f"({intermediate_hidden}, {expected_l2_last_dim}) "
                f"({'packed INT4 via .view(torch.int32)' if is_a8w4_int else 'unpacked'}) "
                f"but got {tuple(l2_weights[0].shape)} with hidden={hidden}."
            )
        if l1_weights[0].dim() < 2:
            raise ValueError(
                "`gmm1_permuted_weight` must contain per-expert 2D tensors in mega_moe "
                "layout [hidden, 2 * intermediate_hidden]."
            )
        if l1_weights[0].shape[-2] != hidden:
            raise ValueError(
                "`gmm1_permuted_weight` must use mega_moe layout "
                "[hidden, 2 * intermediate_hidden] for the mega_moe backend, "
                f"but got first local expert shape {tuple(l1_weights[0].shape)} "
                f"with hidden={hidden}."
            )
        if l1_weights[0].shape[-1] != expected_l1_last_dim:
            raise ValueError(
                "`gmm1_permuted_weight` has an invalid mega_moe layout for "
                f"{inferred_scene}. Expected first local expert shape "
                f"({hidden}, {expected_l1_last_dim}) "
                f"({'packed INT4 via .view(torch.int32)' if is_a8w4_int else 'unpacked'}) "
                f"but got {tuple(l1_weights[0].shape)}."
            )
        return intermediate_hidden

    def _build_runtime_inputs(
        self,
        *,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
    ):
        return self._pad_mega_moe_inputs(
            x,
            topk_idx,
            topk_weights,
            num_max_dispatch_tokens_per_rank,
        )

    def _execute_mega_moe(
        self,
        *,
        mega_moe,
        x_padded: torch.Tensor,
        topk_idx_padded: torch.Tensor,
        topk_weights_padded: torch.Tensor,
        l1_weights,
        l2_weights,
        sym_buffer,
        l1_weights_sf,
        l2_weights_sf,
        l1_bias_list,
        l2_bias_list,
        x_active_mask: torch.Tensor,
        activation: str,
        activation_clamp: Optional[float],
        beta: float,
        linear_beta: Optional[float],
    ):
        return mega_moe(
            x=x_padded,
            topk_ids=topk_idx_padded,
            topk_weights=topk_weights_padded,
            l1_weights=l1_weights,
            l2_weights=l2_weights,
            sym_buffer=sym_buffer,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
            l1_bias=l1_bias_list,
            l2_bias=l2_bias_list,
            x_active_mask=x_active_mask,
            activation=activation,
            activation_clamp=activation_clamp,
            beta=beta,
            linear_beta=linear_beta,
        )

    def run(
        self,
        *,
        buffer,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        quant_mode: int,
        fuse_mode,
        activation: str,
        activation_clamp: Optional[float],
        beta: float,
        linear_beta: Optional[float],
        l1_bias,
        l2_bias,
        dispatch_quant_mode: Optional[int],
        dispatch_quant_out_dtype: Optional[torch.dtype],
        max_recv_token_num: int,
    ):
        _, mega_moe = self._require_mega_moe_ops()
        activation_clamp = buffer._validate_activation_clamp(activation_clamp)
        if activation not in ("swiglu", "swiglu_gpt_oss", "situ"):
            raise ValueError(
                f"Unsupported mega_moe activation {activation!r}. Expected one of "
                "`swiglu`, `swiglu_gpt_oss`, or `situ`."
            )
        if activation != "situ" and not buffer._is_default_beta(beta):
            raise ValueError('`beta` is only valid when `activation="situ"`.')
        if activation != "situ" and not buffer._is_zero_like_linear_beta(linear_beta):
            raise ValueError('`linear_beta` is only valid when `activation="situ"`.')
        if fuse_mode != buffer.FuseMode.FUSED_DEEP_MOE:
            raise NotImplementedError(
                "The mega_moe backend only supports FuseMode.FUSED_DEEP_MOE."
            )

        expected_num_local_experts = num_experts // buffer.group_size
        if expected_num_local_experts * buffer.group_size != num_experts:
            raise ValueError(
                "`num_experts` must be divisible by the process-group size when "
                "using the mega_moe backend."
            )

        (
            l1_weights,
            l2_weights,
            l1_weights_sf,
            l2_weights_sf,
            l1_bias_list,
            l2_bias_list,
            resolved_dispatch_quant_mode,
            resolved_dispatch_quant_out_dtype,
        ) = self._prepare_quant_scene(
            gmm1_permuted_weight=gmm1_permuted_weight,
            gmm1_permuted_weight_scale=gmm1_permuted_weight_scale,
            gmm2_weight=gmm2_weight,
            gmm2_weight_scale=gmm2_weight_scale,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            expected_num_local_experts=expected_num_local_experts,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

        hidden = x.size(1)
        intermediate_hidden = self._validate_weight_layout(
            l1_weights=l1_weights,
            l2_weights=l2_weights,
            hidden=hidden,
            resolved_dispatch_quant_mode=resolved_dispatch_quant_mode,
            l1_bias_list=l1_bias_list,
            l2_bias_list=l2_bias_list,
        )

        sym_buffer = self._get_or_create_mega_moe_symm_buffer(
            group=buffer.group,
            num_experts=num_experts,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_topk=topk_idx.size(1),
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=resolved_dispatch_quant_mode,
            dispatch_quant_out_dtype=resolved_dispatch_quant_out_dtype,
            activation=activation,
        )

        (
            x_padded,
            topk_idx_padded,
            topk_weights_padded,
            x_active_mask,
            original_num_tokens,
        ) = self._build_runtime_inputs(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        )

        output, expert_token_num = self._execute_mega_moe(
            mega_moe=mega_moe,
            x_padded=x_padded,
            topk_idx_padded=topk_idx_padded,
            topk_weights_padded=topk_weights_padded,
            l1_weights=l1_weights,
            l2_weights=l2_weights,
            sym_buffer=sym_buffer,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
            l1_bias_list=l1_bias_list,
            l2_bias_list=l2_bias_list,
            x_active_mask=x_active_mask,
            activation=activation,
            activation_clamp=activation_clamp,
            beta=beta,
            linear_beta=linear_beta,
        )
        return output[:original_num_tokens], expert_token_num
