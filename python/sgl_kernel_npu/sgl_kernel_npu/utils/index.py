import torch
import torch.nn.functional as F


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    num_chunks = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    indices = torch.cat(
        [
            torch.arange(n)
            for n in num_chunks
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
