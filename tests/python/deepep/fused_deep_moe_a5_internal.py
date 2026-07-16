#
# SPDX-License-Identifier: MIT
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Description: Example for fused_deep_moe operator.
# This sample gives an example for using FusedDeepMoe operator and "small" operators, 
# where "small" means the operators taking the same effect with FusedDeepMoe Operator but
# using small operator combination (dispatch + gmm1 + swiglu + gmm2 + combine).
# Create: 2025-12-11
# Note:
# History: 2025-12-11 create example file
#

"""A5-only FusedDeepMoe internal OPP coverage.

Run explicitly on Ascend 950/C310 with CANN 9.0 after installing the DeepEP
custom OPP. The public Buffer API intentionally does not expose these inputs.
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import torchair
import time

import umdk_cam_op_lib

torch.manual_seed(42)
torch_npu.npu.config.allow_internal_format = True
LOG_NAME = "fused_deep_moe_sample_logs"
BASE_KWARGS = {
    "batch_size": 64,
    "token_hidden_size": 7168,
    "moe_intermediate_size": 2048,
    "ep_world_size": 16,
    "moe_expert_num": 64,
    "top_k": 8,
    "test_bfloat16": True,
    "enable_dynamic_bs": False,
    "test_graph": False,
    "with_mc2_mask": False,
    "dynamic_eplb": False,
    "with_share": False,
    "with_smooth": False,
    "enable_nz": False,
    "share_expert_intermediate_size": 2048
}

debug = False
once = False
balance = True

def redirect_output(log_file_path):
    log_path = Path(LOG_NAME) / log_file_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOG_NAME + "/" + log_file_path, "w")
    os.dup2(f.fileno(), sys.stdout.fileno())
    os.dup2(f.fileno(), sys.stderr.fileno())
    return f

def output_to_file(rank_id):
    return rank_id > 0

def convert_tensor_into_parameter(tensor, trans_nz=False):
    if tensor is None:
        return None
    if trans_nz:
        tensor = torch_npu.npu_format_cast(tensor, torch_npu.Format.FRACTAL_NZ)
    return torch.nn.Parameter(tensor, requires_grad=False)

class DecodeMoeOps(torch.nn.Module):

    def __init__(self,
                 ep_hcomm_info,
                 meta_info,
                 weight_datas,
                 share_weight_datas):
        super().__init__()
        self.ep_hcomm_info = ep_hcomm_info
        batch_size, ep_world_size, moe_expert_num, global_rank_id, dynamic_eplb, enable_nz = meta_info
        self.ep_world_size = ep_world_size
        self.moe_expert_num = moe_expert_num
        self.global_rank_id = global_rank_id
        self.dynamic_eplb = dynamic_eplb
        self.enable_nz = enable_nz
        self.global_batch_size = batch_size * ep_world_size
        self.with_share = None
        self.with_smooth = None
        self._checkout_datas(weight_datas, share_weight_datas)
        self._process_share_weights_after_loading(share_weight_datas)
        self._process_weights_after_loading(weight_datas)

    def _checkout_datas(self, weight_datas, share_weight_datas):
        gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, smooth_scales = weight_datas
        share_mm1_weight, share_mm1_weight_scale, share_mm2_weight, share_mm2_weight_scale, share_smooth_scales = share_weight_datas
        if share_mm1_weight is not None:
            assert share_mm1_weight_scale is not None, "share expert need share_mm1_weight_scale"
            assert share_mm2_weight is not None, "share expert need share_mm2_weight"
            assert share_mm2_weight_scale is not None, "share expert need share_mm2_weight_scale"
            if smooth_scales is not None:
                assert share_smooth_scales is not None, "share expert need share_smooth_scales"
                self.with_smooth = True
            else:
                self.with_smooth = False
            self.with_share = True
        else:
            self.with_share = False

    def _process_share_weights_after_loading(self, share_weight_datas):
        share_gmm1_weight, share_gmm1_weight_scale, share_gmm2_weight, share_gmm2_weight_scale, share_smooth_scales = share_weight_datas
        self.share_gmm1_weight = convert_tensor_into_parameter(share_gmm1_weight, trans_nz=self.enable_nz)
        self.share_gmm1_weight_scale = convert_tensor_into_parameter(share_gmm1_weight_scale)
        self.share_gmm2_weight = convert_tensor_into_parameter(share_gmm2_weight, trans_nz=self.enable_nz)
        self.share_gmm2_weight_scale = convert_tensor_into_parameter(share_gmm2_weight_scale)
        self.share_smooth_scales = convert_tensor_into_parameter(share_smooth_scales)

    def _process_weights_after_loading(self, weight_datas):
        gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, smooth_scales = weight_datas
        self.gmm1_weight = convert_tensor_into_parameter(gmm1_weight, trans_nz=self.enable_nz)
        self.gmm1_weight_scale = convert_tensor_into_parameter(gmm1_weight_scale)
        self.gmm2_weight = convert_tensor_into_parameter(gmm2_weight, trans_nz=self.enable_nz)
        self.gmm2_weight_scale = convert_tensor_into_parameter(gmm2_weight_scale)
        self.smooth_scales = convert_tensor_into_parameter(smooth_scales)

    def _apply_ops(self, x, expert_ids, expert_scales, x_active_mask):
        raise NotImplementedError("To be implemented in subclass")

    def forward(self, x, expert_ids, expert_scales, x_active_mask):
        return self._apply_ops(x, expert_ids, expert_scales, x_active_mask)


class SmallOps(DecodeMoeOps):

    def __init__(self,
                 ep_hcomm_info,
                 meta_info,
                 weight_datas,
                 share_weight_datas):
        super().__init__(ep_hcomm_info, meta_info, weight_datas, share_weight_datas)
        self.shared_expert_rank_num = 0
        self.tp_hcomm_info = ""

    def share_compute(self, x):
        output_dtype = x.dtype
        x1, x1_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        x1_scale = x1_scale.view(torch.float8_e8m0fnu)
        y1_fp = torch_npu.npu_quant_matmul(x1, self.share_gmm1_weight, self.share_gmm1_weight_scale, pertoken_scale=x1_scale, output_dtype=output_dtype)
        swiglu_out = torch_npu.npu_swiglu(y1_fp)
        x2, x2_scale = torch_npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)
        x2_scale = x2_scale.view(torch.float8_e8m0fnu)
        y2_fp = torch_npu.npu_quant_matmul(x2, self.share_gmm2_weight, self.share_gmm2_weight_scale, pertoken_scale=x2_scale, output_dtype=output_dtype)
        if debug and (self.global_rank_id == 0 or True):
            # tmp tensor
            torch.save(x1.cpu(), f"rank_{self.global_rank_id}_small_share_x1.pt")
            torch.save(x1_scale.view(torch.uint8).cpu(), f"rank_{self.global_rank_id}_small_share_x1_scale.pt")
            torch.save(y1_fp.cpu(), f"rank_{self.global_rank_id}_small_share_y1_fp.pt")
            torch.save(swiglu_out.cpu(), f"rank_{self.global_rank_id}_small_share_swiglu_out.pt")
            torch.save(x2.cpu(), f"rank_{self.global_rank_id}_small_share_x2.pt")
            torch.save(x2_scale.view(torch.uint8).cpu(), f"rank_{self.global_rank_id}_small_share_x2_scale.pt")
            # out tensor
            torch.save(y2_fp.cpu(), f"rank_{self.global_rank_id}_small_share_share_output.pt")
        return y2_fp

    def _apply_ops(self, x, expert_ids, expert_scales, x_active_mask):
        if self.with_share:
            share_output = self.share_compute(x)
        else:
            share_output = None
        outputs = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            scales=self.smooth_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            group_tp=self.tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            quant_mode=4,
            global_bs=self.global_batch_size,
            expert_token_nums_type=1,  # 0代表前缀和，1代表各自数量
            y_dtype=torch.float8_e4m3fn,
            # scales_dtype=torch.float8_e8m0fnu
        )
        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_send_counts, tp_send_counts, expand_scales = outputs
        output_dtype = x.dtype
        dynamic_scales = dynamic_scales.view(*(dynamic_scales.shape[:-1]), -1, 2).view(torch.float8_e8m0fnu)

        y1_fp = torch_npu.npu_grouped_matmul(
            x=[expand_x],
            weight=[self.gmm1_weight],
            scale=[self.gmm1_weight_scale],
            per_token_scale=[dynamic_scales],
            split_item=2,
            group_list_type=1,  # 默认为0，代表前缀和形式
            group_type=0,  # 0代表m轴分组
            group_list=expert_token_nums,
            output_dtype=output_dtype)[0]
        swiglu_out = torch_npu.npu_swiglu(y1_fp)
        x2, x2_scale = torch_npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)
        x2_scale = x2_scale.view(torch.float8_e8m0fnu)
        y2_fp = torch_npu.npu_grouped_matmul(x=[x2],
                                          weight=[self.gmm2_weight],
                                          scale=[self.gmm2_weight_scale],
                                          per_token_scale=[x2_scale],
                                          split_item=2,
                                          group_list_type=1,
                                          group_type=0,
                                          group_list=expert_token_nums,
                                          output_dtype=output_dtype)[0]
        combine_output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=y2_fp,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            tp_send_counts=tp_send_counts,
            expand_scales=expand_scales,
            group_tp=self.tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            global_bs=self.global_batch_size)
        if debug and (self.global_rank_id == 0 or True):
            # tmp tensor
            torch.save(expand_x.cpu(), f"rank_{self.global_rank_id}_small_x1.pt")
            torch.save(dynamic_scales.view(torch.uint8).cpu(), f"rank_{self.global_rank_id}_small_x1_scale.pt")
            torch.save(y1_fp.cpu(), f"rank_{self.global_rank_id}_small_y1_fp.pt")
            torch.save(swiglu_out.cpu(), f"rank_{self.global_rank_id}_small_swiglu_out.pt")
            torch.save(x2.cpu(), f"rank_{self.global_rank_id}_small_x2.pt")
            torch.save(x2_scale.view(torch.uint8).cpu(), f"rank_{self.global_rank_id}_small_x2_scale.pt")
            torch.save(y2_fp.cpu(), f"rank_{self.global_rank_id}_small_y2_fp.pt")
            torch.save(ep_send_counts.cpu(), f"rank_{self.global_rank_id}_small_ep_send_counts.pt")
            # out tensor
            torch.save(combine_output.cpu(), f"rank_{self.global_rank_id}_small_moe_output.pt")
            torch.save(expert_token_nums.cpu(), f"rank_{self.global_rank_id}_small_group_list.pt")
        return (combine_output, share_output, expert_token_nums)


class FusionOp(DecodeMoeOps):

    def __init__(self,
                 ep_hcomm_info,
                 meta_info,
                 weight_datas,
                 share_weight_datas):
        super().__init__(ep_hcomm_info, meta_info, weight_datas, share_weight_datas)

    def _apply_ops(self, x, expert_ids, expert_scales, x_active_mask):
        if debug and (self.global_rank_id == 0 or True):
            self.share_smooth_scales_fp32 = torch.zeros(64 * 1024 * 1024).npu()
        output, share_output, expert_token_nums = torch.ops.hwcomputing.fused_deep_moe(
            x=x,
            expert_ids=expert_ids,
            gmm1_weight=self.gmm1_weight,
            gmm1_weight_scale=self.gmm1_weight_scale,
            gmm2_weight=self.gmm2_weight,
            gmm2_weight_scale=self.gmm2_weight_scale,
            expert_scales=expert_scales,
            share_gmm1_weight=None if self.share_gmm1_weight is None else self.share_gmm1_weight.view(torch.int8),
            share_gmm1_weight_scale=None if self.share_gmm1_weight is None else self.share_gmm1_weight_scale.view(torch.int8),
            share_gmm2_weight=None if self.share_gmm1_weight is None else self.share_gmm2_weight.view(torch.int8),
            share_gmm2_weight_scale=None if self.share_gmm1_weight is None else self.share_gmm2_weight_scale.view(torch.int8),
            expert_smooth_scales=self.smooth_scales,
            share_smooth_scales=self.share_smooth_scales_fp32,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_rank_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            quant_mode=0,
            global_bs=self.global_batch_size)
        if debug and (self.global_rank_id == 0 or True):
            torch.save(self.share_smooth_scales_fp32.cpu().view(torch.int8), f"rank_{self.global_rank_id}_fused_dfx_tensor.pt")
            torch.save(output.cpu(), f"rank_{self.global_rank_id}_fused_moe_output.pt")
            torch.save(expert_token_nums.cpu(), f"rank_{self.global_rank_id}_fused_group_list.pt")
        return (output, share_output, expert_token_nums)

    def _process_share_weights_after_loading(self, share_weight_datas):
        super()._process_share_weights_after_loading(share_weight_datas)
        _, _, _, _, share_smooth_scales = share_weight_datas
        if self.with_share and self.with_smooth:
            self.share_smooth_scales_fp32 = convert_tensor_into_parameter(share_smooth_scales.float())
        else:
            self.share_smooth_scales_fp32 = None

    def _process_weights_after_loading(self, weight_datas):
        gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, smooth_scales = weight_datas
        gmm1_weight = convert_tensor_into_parameter(gmm1_weight, trans_nz=self.enable_nz)
        gmm1_weight_scale = convert_tensor_into_parameter(gmm1_weight_scale)
        gmm2_weight = convert_tensor_into_parameter(gmm2_weight, trans_nz=self.enable_nz)
        gmm2_weight_scale = convert_tensor_into_parameter(gmm2_weight_scale)
        if self.dynamic_eplb:
            self.gmm1_weight = [
                weight.clone().view(torch.int8) for weight in gmm1_weight.unbind(dim=0)
            ]
            self.gmm1_weight_scale = [
                weight.clone().view(torch.int8) for weight in gmm1_weight_scale.unbind(dim=0)
            ]
            self.gmm2_weight = [
                weight.clone().view(torch.int8) for weight in gmm2_weight.unbind(dim=0)
            ]
            self.gmm2_weight_scale = [
                weight.clone().view(torch.int8) for weight in gmm2_weight_scale.unbind(dim=0)
            ]
        else:
            # self.gmm1_weight = [gmm1_weight.clone()]
            # self.gmm1_weight_scale = [gmm1_weight_scale.clone()]
            # self.gmm2_weight = [gmm2_weight.clone()]
            # self.gmm2_weight_scale = [gmm2_weight_scale.clone()]
            self.gmm1_weight = [gmm1_weight.view(torch.int8)]
            self.gmm1_weight_scale = [gmm1_weight_scale.view(torch.int8)]
            self.gmm2_weight = [gmm2_weight.view(torch.int8)]
            self.gmm2_weight_scale = [gmm2_weight_scale.view(torch.int8)]
        self.smooth_scales = convert_tensor_into_parameter(smooth_scales)

def generate_datas(batch_size,
                   token_hidden_size,
                   moe_intermediate_size,
                   ep_world_size,
                   moe_expert_num,
                   global_rank_id,
                   top_k=8,
                   test_bfloat16=True,
                   enable_dynamic_bs=False,
                   with_mc2_mask=False,
                   with_share=False,
                   with_smooth=False,
                   share_expert_intermediate_size=None):
    moe_expert_num_per_rank = moe_expert_num // ep_world_size
    actual_bs = int(
        torch.randint(2 if with_mc2_mask else 1, batch_size, [1]).item(
        ) if enable_dynamic_bs else batch_size)
    local_expert_num = moe_expert_num_per_rank
    gmm1_input_dim = token_hidden_size
    gmm1_output_dim = moe_intermediate_size * 2
    gmm2_input_dim = moe_intermediate_size
    gmm2_output_dim = token_hidden_size
    x = torch.rand([actual_bs, token_hidden_size]) * 10 - 5

    # random expert_ids
    score = torch.rand(batch_size, moe_expert_num)
    expert_ids = torch.topk(score, k=top_k)[1].to(torch.int32)

    # Absolute uniform expert_ids
    if balance:
        expert_ids = torch.arange(
            global_rank_id * batch_size * top_k,
            global_rank_id * batch_size * top_k + actual_bs * top_k).to(
                torch.int32).view(actual_bs, top_k)
        expert_ids = expert_ids % moe_expert_num

    gmm1_weight_bf16 = torch.rand([local_expert_num, gmm1_input_dim, gmm1_output_dim]).bfloat16().npu() * 2 - 1
    gmm1_weight, gmm1_weight_scale = torch_npu.npu_dynamic_mx_quant(gmm1_weight_bf16, dst_type=torch.float8_e4m3fn, axis=1)
    gmm1_weight_scale = gmm1_weight_scale.view(torch.float8_e8m0fnu)
    gmm2_weight_bf16 = torch.rand([local_expert_num, gmm2_input_dim, gmm2_output_dim]).bfloat16().npu() * 2 - 1
    gmm2_weight, gmm2_weight_scale = torch_npu.npu_dynamic_mx_quant(gmm2_weight_bf16, dst_type=torch.float8_e4m3fn, axis=1)
    gmm2_weight_scale = gmm2_weight_scale.view(torch.float8_e8m0fnu)

    expert_scales = torch.rand(actual_bs, top_k)
    # Generate shared expert weights
    share_mm1_weight = None
    share_mm1_weight_scale = None
    share_mm2_weight = None
    share_mm2_weight_scale = None
    if with_share:
        # Use share_expert_intermediate_size for shared expert gmm1HLen
        share_gmm2_input_dim = share_expert_intermediate_size if share_expert_intermediate_size is not None else moe_intermediate_size
        share_gmm1_output_dim = share_gmm2_input_dim * 2
        share_mm1_weight_bf16 = torch.rand([gmm1_input_dim, share_gmm1_output_dim]).bfloat16().npu() * 2 - 1
        share_mm1_weight, share_mm1_weight_scale = torch_npu.npu_dynamic_mx_quant(share_mm1_weight_bf16, dst_type=torch.float8_e4m3fn, axis=0)
        share_mm1_weight_scale = share_mm1_weight_scale.view(torch.float8_e8m0fnu)
        share_mm2_weight_bf16 = torch.rand([share_gmm2_input_dim, gmm2_output_dim]).bfloat16().npu() * 2 - 1
        share_mm2_weight, share_mm2_weight_scale = torch_npu.npu_dynamic_mx_quant(share_mm2_weight_bf16, dst_type=torch.float8_e4m3fn, axis=0)
        share_mm2_weight_scale = share_mm2_weight_scale.view(torch.float8_e8m0fnu)

    if test_bfloat16:
        x = x.bfloat16()
    else:
        x = x.half() / 10
    smooth_scales = None
    share_smooth_scales = None
    if with_smooth:
        smooth_scales = torch.rand([moe_expert_num, token_hidden_size])
        share_smooth_scales = torch.rand([token_hidden_size]).to(x.dtype)
    x_active_mask = None
    valid_token_num = actual_bs
    if with_mc2_mask:
        valid_token_num = int(torch.randint(1, actual_bs, [1]).item())
        x_active_mask = torch.cat(
            (torch.ones(valid_token_num),
             torch.zeros(actual_bs - valid_token_num))).bool()
    return (x, expert_ids, expert_scales, x_active_mask), \
            (gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, smooth_scales), \
            (share_mm1_weight, share_mm1_weight_scale, share_mm2_weight, share_mm2_weight_scale, share_smooth_scales), \
            actual_bs, valid_token_num


def run_once(local_rank_id,
             batch_size,
             token_hidden_size,
             moe_intermediate_size,
             ep_world_size,
             moe_expert_num,
             top_k=8,
             test_bfloat16=True,
             enable_dynamic_bs=False,
             test_graph=False,
             with_mc2_mask=False,
             dynamic_eplb=False,
             with_share=False,
             with_smooth=False,
             enable_nz=False,
             share_expert_intermediate_size=None):
    # 配置日志输出文件名
    log_file = redirect_output(f"local_rank_{local_rank_id}.log"
                               ) if output_to_file(local_rank_id) else None
    # 使用A3 单机16DIE进行测试
    global_rank_id = local_rank_id
    device_id = local_rank_id % 16
    torch_npu.npu.set_device(device_id)

    # 初始化分布式环境
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "27500"  # 端口号随意
    dist.init_process_group(backend="hccl",
                            rank=local_rank_id,
                            world_size=ep_world_size)
    ep_ranks_list = list(np.arange(0, ep_world_size))
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    ep_group_small = dist.new_group(backend="hccl", ranks=ep_ranks_list)

    ep_hcomm_info_fused = ep_group._get_backend(
        torch.device("npu")).get_hccl_comm_name(local_rank_id)
    ep_hcomm_info_small = ep_group_small._get_backend(
        torch.device("npu")).get_hccl_comm_name(local_rank_id)
    torch_npu.npu.synchronize(device_id)

    # 构建必要参数和权重数据
    parameter = (batch_size, token_hidden_size, moe_intermediate_size,
                 ep_world_size, moe_expert_num, global_rank_id, top_k,
                 test_bfloat16, enable_dynamic_bs, with_mc2_mask,
                 with_share, with_smooth, share_expert_intermediate_size)
    input_datas, weight_datas, share_weight_datas, actual_bs, valid_token_num = generate_datas(*parameter)
    input_datas = [
        data.npu() if data is not None else None for data in input_datas
    ]
    meta_info = (batch_size, ep_world_size, moe_expert_num, global_rank_id, dynamic_eplb, enable_nz)
    weight_datas = [
        data.npu() if data is not None else None for data in weight_datas
    ]
    share_weight_datas = [
        data.npu() if data is not None else None for data in share_weight_datas
    ]
    
    small_ops = SmallOps(ep_hcomm_info_small, meta_info, weight_datas, share_weight_datas).npu()  # type: ignore
    fused_ops = FusionOp(ep_hcomm_info_fused, meta_info, weight_datas, share_weight_datas).npu()  # type: ignore
    if test_graph:
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        fused_ops = torch.compile(fused_ops, backend=npu_backend)
    warm_iter = 0 if (debug or once) else 10
    test_iter = 1 if (debug or once) else 100

    for _ in range(warm_iter):
        small_op_output = small_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    print("small warmup end")
    start = time.time()
    for _ in range(test_iter):
        small_op_output = small_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    print("small end " + f"test_iter avg time= {((time.time() - start) * 1000 * 1000 / test_iter):.1f}us")
    for _ in range(warm_iter):
        fused_op_output = fused_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    print("fused warmup end")
    start = time.time()
    for _ in range(test_iter):
        fused_op_output = fused_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    print("fused end " + f"test_iter avg time= {((time.time() - start) * 1000 * 1000 / test_iter):.1f}us")
    small_op_token_output, small_op_share_output, small_op_count_output = small_op_output
    fused_op_token_output, fused_op_share_output, fused_op_count_output = fused_op_output
    
    torch_npu.npu.synchronize(device_id)

    # 处理资源销毁
    dist.destroy_process_group()
    if log_file is not None:
        log_file.close()
    try:
        ...
        print(f"routed compare begin, {valid_token_num= }")
        torch.testing.assert_close(small_op_token_output.cpu()[:valid_token_num],
                                fused_op_token_output.cpu()[:valid_token_num],
                                atol=2.0,
                                rtol=0.02)
        torch.testing.assert_close(small_op_count_output.cpu(),
                                fused_op_count_output.cpu())
        print("routed compare pass")
        if with_share:
            torch.testing.assert_close(small_op_share_output.cpu(),
                                    fused_op_share_output.cpu(),
                                    atol=2.0,
                                    rtol=0.02)
            print("shared compare pass")
    except Exception as e:
        print(f"rank-{global_rank_id} Failed!, message is {e}")
        
    else:
        print(f"rank-{global_rank_id} Passed!")
    finally:
        ...
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_fused_deep_moe_base():
    custom_kwargs = BASE_KWARGS
    ep_world_size = custom_kwargs["ep_world_size"]
    custom_args = tuple(custom_kwargs.values())
    mp.spawn(run_once, args=custom_args, nprocs=ep_world_size, join=True)


@torch.inference_mode()
def test_fused_deep_moe_with_mc2_mask():
    custom_kwargs = BASE_KWARGS
    custom_kwargs["with_mc2_mask"] = True
    ep_world_size = custom_kwargs["ep_world_size"]
    custom_args = tuple(custom_kwargs.values())
    mp.spawn(run_once, args=custom_args, nprocs=ep_world_size, join=True)


@torch.inference_mode()
def test_fused_deep_moe_eplb():
    custom_kwargs = BASE_KWARGS
    custom_kwargs["dynamic_eplb"] = True
    ep_world_size = custom_kwargs["ep_world_size"]
    custom_args = tuple(custom_kwargs.values())
    mp.spawn(run_once, args=custom_args, nprocs=ep_world_size, join=True)


@torch.inference_mode()
def test_fused_deep_moe_shared_smooth_nz():
    custom_kwargs = dict(BASE_KWARGS)
    custom_kwargs["with_share"] = True
    custom_kwargs["with_smooth"] = True
    custom_kwargs["enable_nz"] = True
    ep_world_size = custom_kwargs["ep_world_size"]
    mp.spawn(run_once, args=tuple(custom_kwargs.values()), nprocs=ep_world_size, join=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--token_hidden_size", type=int, default=7168)
    parser.add_argument("--moe_intermediate_size", type=int, default=2048)
    parser.add_argument("--ep_world_size", type=int, default=4)
    parser.add_argument("--moe_expert_num", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--test_float16", action="store_true", default=False)
    parser.add_argument("--enable_dynamic_bs", action="store_true", default=False)
    parser.add_argument("--test_graph", action="store_true", default=False)
    parser.add_argument("--with_mc2_mask", action="store_true", default=False)
    parser.add_argument("--dynamic_eplb", action="store_true", default=False)
    parser.add_argument("--with_share", action="store_true", default=False)
    parser.add_argument("--with_smooth", action="store_true", default=False)
    parser.add_argument("--enable_nz", action="store_true", default=False)
    parser.add_argument("--all-features", action="store_true", default=False)
    parser.add_argument("--share_expert_intermediate_size", type=int)
    args = parser.parse_args()
    BASE_KWARGS["batch_size"] = args.batch_size
    BASE_KWARGS["token_hidden_size"] = args.token_hidden_size
    BASE_KWARGS["moe_intermediate_size"] = args.moe_intermediate_size
    BASE_KWARGS["moe_expert_num"] = args.moe_expert_num
    BASE_KWARGS["ep_world_size"] = args.ep_world_size
    BASE_KWARGS["top_k"] = args.top_k
    BASE_KWARGS["test_bfloat16"] = not args.test_float16
    BASE_KWARGS["enable_dynamic_bs"] = args.enable_dynamic_bs
    BASE_KWARGS["test_graph"] = args.test_graph
    BASE_KWARGS["with_mc2_mask"] = args.with_mc2_mask
    BASE_KWARGS["dynamic_eplb"] = args.dynamic_eplb
    BASE_KWARGS["with_share"] = args.with_share
    BASE_KWARGS["with_smooth"] = args.with_smooth
    BASE_KWARGS["enable_nz"] = args.enable_nz
    BASE_KWARGS["share_expert_intermediate_size"] = args.share_expert_intermediate_size \
        if args.share_expert_intermediate_size is not None else args.moe_intermediate_size
    print(f"{BASE_KWARGS=}")
    test_fused_deep_moe_base()
    if args.all_features:
        test_fused_deep_moe_with_mc2_mask()
        test_fused_deep_moe_eplb()
        test_fused_deep_moe_shared_smooth_nz()
