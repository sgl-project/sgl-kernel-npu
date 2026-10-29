"""Independent reference for the single-request A5 decode metadata layout.

The general host scheduler balances variable-length rows across cube cores.
For one decode query, all work belongs to the first logical core; the 128-head
layout duplicates that core's entry and records its S2 tile bound in every
physical cube slot. This reference deliberately supports only that simple
schedule, so it cannot silently be used as a general scheduler.
"""

import torch


def kv_quant_sparse_attn_sharedkv_metadata(**kwargs):
    heads = kwargs["num_heads_q"]
    layout = kwargs["layout_q"]
    cu_q = kwargs["cu_seqlens_q"]
    kv_lengths = kwargs["seqused_kv"]
    batch_size = kwargs["batch_size"]
    if batch_size == 0:
        if cu_q is not None:
            batch_size = cu_q.numel() - 1
        elif kv_lengths is not None:
            batch_size = kv_lengths.numel()

    if (
        heads not in (64, 128)
        or kwargs["num_heads_kv"] != 1
        or kwargs["head_dim"] != 512
        or layout not in ("BSND", "TND")
        or kwargs["layout_kv"] != "PA_ND"
        or batch_size != 1
        or kv_lengths is None
        or kv_lengths.device.type != "cpu"
        or kv_lengths.dtype != torch.int32
        or kv_lengths.numel() != 1
        or int(kv_lengths[0]) <= 0
        or kwargs["max_seqlen_q"] != 1
        or not kwargs["has_ori_kv"]
        or kwargs["ori_mask_mode"] != 4
        or kwargs["cmp_mask_mode"] != 3
        or kwargs["ori_win_left"] != 127
        or kwargs["ori_win_right"] != 0
        or kwargs["kv_quant_mode"] != 1
        or kwargs["tile_size"] != 64
        or kwargs["rope_head_dim"] != 64
        or (heads == 128 and kwargs["has_cmp_kv"])
    ):
        raise NotImplementedError(
            "torch_native metadata supports one uncompressed decode query "
            "with positive KV length (64 heads may also use compressed KV)"
        )
    if layout == "TND" and (
        cu_q is None
        or cu_q.device.type != "cpu"
        or cu_q.dtype != torch.int32
        or cu_q.tolist() != [0, 1]
    ):
        raise NotImplementedError("TND reference requires cu_seqlens_q=[0, 1]")
    if kwargs["has_cmp_kv"] and (
        kwargs["cmp_ratio"] not in (4, 128) or kwargs["cmp_topk"] not in (0, 512, 1024)
    ):
        raise NotImplementedError("Unsupported compressed-KV mode")

    schedule = torch.zeros(1024, dtype=torch.int32, device="cpu")
    schedule[0] = 1  # First cube core enabled.
    schedule[4] = 1  # Exclusive batch/head-group end.
    if heads == 128:
        props = torch.npu.get_device_properties(torch.npu.current_device())
        cube_cores = int(props.cube_core_num)
        if cube_cores < 2 or cube_cores > 36 or cube_cores % 2:
            raise NotImplementedError(
                "128-head reference requires an even cube-core count"
            )
        schedule[9] = 1  # Paired physical core runs the same logical work.
        schedule[13] = 1
        schedule[8 : 9 * cube_cores : 9] = 1  # One S2 tile per physical core.
    return schedule.to("npu")
