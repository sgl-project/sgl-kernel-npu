import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    global_rank = node_rank * num_local_ranks + local_rank
    world_size = num_nodes * num_local_ranks

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=global_rank,
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    group = dist.new_group(list(range(world_size)))

    return dist.get_rank(), dist.get_world_size(), group


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    device = torch.device("npu")
    torch.npu.synchronize()

    # Flush L2 cache with 256 MB data
    cache = torch.empty(int(256e6 // 4), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2 cache
    cache.zero_()
    torch.npu.synchronize()

    # Timing
    times = []
    for _ in range(num_tests):
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        if post_fn is not None:
            post_fn()

        torch.npu.synchronize()
        elapsed_time = start.elapsed_time(end) / 1e3  # ms -> s
        times.append(elapsed_time)

    times = np.array(times[1:])  # Remove the first timing
    return np.average(times), np.min(times), np.max(times)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int8).sum().item()


def diagnose_matrix(
    mat,
    thres_col=3.0,
    thres_row=3.0,
    thres_point=5.0,
    suppress_points_in_strong_rowscols=True,
):
    """
    Detect abnormal columns, rows, and individual points in a 2D wait-time matrix.
    Arguments:
        mat (np.ndarray): 2D array where mat[i, j] is the waiting time of source i for destination j to
            receive(dispatch)/send(combine) the token
        thres_col/thres_row/thres_point(float): The ratio of the average waiting time for abnormal rank
            to the average waiting time for all ranks
        suppress_points_in_strong_rowscols (bool): If True, exclude points already in detected abnormal
            rows/columns.
    Returns:
        dict: {
            "abnormal_cols": List[List[int, float, float]],  # abnormal column indices
            "abnormal_rows": List[List[int, float, float]],  # abnormal row indices
            "abnormal_points": List[List[int, int, float, float]]  # abnormal points
        }
    """
    mat = mat.cpu().numpy()
    # 1. Check for abnormal columns
    col_means = mat.mean(axis=0)
    z_col = col_means / (col_means.mean() + 1e-8)
    abnormal_cols = [
        [j, col_means[j], z_col[j]] for j in np.where(z_col > thres_col)[0]
    ]

    # 2. Check for abnormal rows
    row_means = mat.mean(axis=1)
    z_row = row_means / (row_means.mean() + 1e-8)
    abnormal_rows = [
        [i, row_means[i], z_row[i]] for i in np.where(z_row > thres_row)[0]
    ]

    # 3. Check for abnormal single points
    z_all = mat / (mat.mean() + 1e-8)
    # Get all positions with z-score > threshold
    abnormal_points = [
        [i, j, mat[i, j], z_all[i, j]]
        for i in range(mat.shape[0])
        for j in range(mat.shape[1])
        if z_all[i, j] > thres_point
    ]
    # Optionally remove points that are in already detected abnormal rows
    # or columns
    if suppress_points_in_strong_rowscols:
        strong_rows = [row[0] for row in abnormal_rows]
        strong_cols = [col[0] for col in abnormal_cols]
        abnormal_points = [
            [i, j, v, z]
            for [i, j, v, z] in abnormal_points
            if i not in strong_rows and j not in strong_cols
        ]
    # 4. Return for automatic processing
    return {
        "abnormal_cols": abnormal_cols,
        "abnormal_rows": abnormal_rows,
        "abnormal_points": abnormal_points,
    }
