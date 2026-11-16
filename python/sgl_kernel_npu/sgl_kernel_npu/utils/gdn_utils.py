"""Prepare chunk metadata for packed GDN sequences."""

import functools
from typing import Any, Callable

import torch


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache the four most recently used argument sets by tensor identity."""

    cache_entries = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(
                    (
                        (a is b)
                        if isinstance(a, torch.Tensor)
                        else (a == b and type(a) == type(b))
                    )
                    for a, b in zip(args, last_args)
                ) and all(
                    (
                        k in last_kwargs and (v is last_kwargs[k])
                        if isinstance(v, torch.Tensor)
                        else (v == last_kwargs[k] and type(v) == type(last_kwargs[k]))
                    )
                    for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    cu_seqlens_i64 = cu_seqlens.to(torch.int64)
    return cu_seqlens_i64[1:] - cu_seqlens_i64[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_counts = (prepare_lens(cu_seqlens) + chunk_size - 1) // chunk_size
    indices = torch.cat([torch.arange(n) for n in chunk_counts.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
