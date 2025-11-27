import inspect
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch_npu


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
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)
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


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch_npu.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], schedule=schedule
        ) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="npu")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="npu")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="npu"))
                for _ in range(num_tests):
                    fn()
                torch.npu.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)

    kernel_names = (kernel_names,) if not is_tuple else kernel_names
    assert all(isinstance(name, str) for name in kernel_names)
    # Expand the kernels by periods

    # If the json file exists, `torch_npu.profiler.export_chrome_trace` will use the append write mode,
    # which will cause problems with the json format, so here we use a random file name instead of creating a temporary file
    temp_path = Path(tempfile.gettempdir()) / f"trace_{uuid.uuid4().hex}.json"
    prof.export_chrome_trace(temp_path)
    profile_data = json.loads(Path(temp_path).read_text())

    # Return average kernel durations
    kernel_durations = []
    for kernel_name in kernel_names:
        events = [event for event in profile_data if kernel_name == event["name"]]
        assert len(events) > 0, f"Kernel '{kernel_name}' not found in trace"
        events = sorted(events, key=lambda event: event["ts"])
        durations = [event["dur"] / 1e6 for event in events]
        if num_kernels_per_period > 1:
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations.append(
                [
                    sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                    for j in range(num_kernels_per_period)
                ]
            )
        else:
            num_kernel_patterns = len(durations)
            kernel_durations.append(sum(durations) / num_kernel_patterns)

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    os.unlink(temp_path)

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


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


def calculate_avg_stats(
    dispatch_t,
    num_dispatch_comm_bytes,
    combine_t,
    num_combine_comm_bytes,
    rank,
    num_ranks,
    root_rank: 0,
):
    # dispatch_t / combine_t: the unit is second
    local_stats = torch.tensor(
        [
            dispatch_t * 1e6,
            num_dispatch_comm_bytes,
            combine_t * 1e6,
            num_combine_comm_bytes,
        ],
        dtype=torch.float64,
        device="npu",
    )
    gather_stats = (
        [torch.zeros_like(local_stats) for _ in range(num_ranks)]
        if rank == root_rank
        else None
    )
    dist.gather(local_stats, gather_list=gather_stats, dst=0)
    if rank == root_rank:
        stats_tensor = torch.stack(gather_stats)  # Shape [num_ranks, 4]
        dispatch_latency = stats_tensor[:, 0]  # us
        dispatch_bytes = stats_tensor[:, 1]  # bytes
        combine_latency = stats_tensor[:, 2]  # us
        combine_bytes = stats_tensor[:, 3]  # bytes

        avg_dispatch_lat = torch.mean(dispatch_latency)
        avg_dispatch_bytes = torch.mean(dispatch_bytes)
        avg_combine_lat = torch.mean(combine_latency)
        avg_combine_bytes = torch.mean(combine_bytes)

        avg_dispatch_bw = avg_dispatch_bytes / avg_dispatch_lat * 1e-3  # GB/s
        avg_combine_bw = avg_combine_bytes / avg_combine_lat * 1e-3  # GB/s
        print(
            f"\n\nAverage Dispatch bandwidth: {avg_dispatch_bw:.2f} GB/s, avg_t={avg_dispatch_lat:.2f} us \n"
            f"Average Combine bandwidth: {avg_combine_bw:.2f} GB/s, avg_t={avg_combine_lat:.2f} us\n\n",
            flush=True,
        )
