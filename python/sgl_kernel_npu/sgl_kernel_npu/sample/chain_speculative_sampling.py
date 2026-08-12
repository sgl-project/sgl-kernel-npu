import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _chain_rejection_accept_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    target_probs,
    draft_probs,
    metadata,
    num_draft_tokens: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    num_draft_prob_rows: tl.constexpr,
    vocab_size: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_offset = req_idx * num_draft_tokens

    cur_prob_row = tl.full((), 0, tl.int64)
    last_accepted_idx = tl.load(retrive_index + row_offset).to(tl.int64)
    num_accepted = 0
    active = tl.full((), 1, tl.int32)

    tl.store(accept_index + req_idx * num_speculative_tokens, last_accepted_idx)

    # Linear Leviathan/Chen verification. Candidate 0 is the root; candidate
    # step uses probability row step - 1 until a rejection terminates the chain.
    for step in range(1, num_draft_tokens):
        if active == 1:
            draft_token = tl.load(candidates + row_offset + step).to(tl.int64)
            target_offset = (row_offset + cur_prob_row) * vocab_size + draft_token
            draft_offset = (
                req_idx * num_draft_prob_rows + cur_prob_row
            ) * vocab_size + draft_token
            target_prob = tl.load(target_probs + target_offset).to(tl.float32)
            draft_prob = tl.load(draft_probs + draft_offset).to(tl.float32)
            coin = tl.load(uniform_samples + row_offset + step - 1).to(tl.float32)

            if coin * draft_prob < target_prob:
                tl.store(predicts + last_accepted_idx, draft_token)
                num_accepted += 1
                draft_idx = tl.load(retrive_index + row_offset + step).to(tl.int64)
                tl.store(
                    accept_index + req_idx * num_speculative_tokens + num_accepted,
                    draft_idx,
                )
                last_accepted_idx = draft_idx
                # Keep this loop-carried value int64 across both branches.
                # Triton infers the constexpr loop variable `step` as int32.
                cur_prob_row = tl.full((), step, tl.int64)
            else:
                active = 0

    tl.store(accept_token_num + req_idx, num_accepted)

    # metadata = [target row, output slot, all drafts accepted].
    metadata_offset = req_idx * 3
    tl.store(metadata + metadata_offset, cur_prob_row)
    tl.store(metadata + metadata_offset + 1, last_accepted_idx)
    tl.store(metadata + metadata_offset + 2, active)


@triton.jit
def _chain_rejection_block_sum_kernel(
    target_probs,
    draft_probs,
    metadata,
    block_sums,
    batch_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    num_draft_prob_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    vocab_block_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_blocks = batch_size * num_vocab_blocks

    for flat_block_idx in tl.range(program_idx, total_blocks, num_programs):
        req_idx = flat_block_idx // num_vocab_blocks
        block_idx = flat_block_idx % num_vocab_blocks
        vocab_offsets = block_idx * vocab_block_size + tl.arange(0, vocab_block_size)
        vocab_mask = vocab_offsets < vocab_size

        metadata_offset = req_idx * 3
        target_row = tl.load(metadata + metadata_offset).to(tl.int64)
        all_accepted = tl.load(metadata + metadata_offset + 2).to(tl.int32)
        target_offset = (req_idx * num_draft_tokens + target_row) * vocab_size
        target = tl.load(
            target_probs + target_offset + vocab_offsets,
            mask=vocab_mask,
            other=0.0,
        ).to(tl.float32)

        if all_accepted == 1:
            residual = target
        else:
            draft_offset = (req_idx * num_draft_prob_rows + target_row) * vocab_size
            draft = tl.load(
                draft_probs + draft_offset + vocab_offsets,
                mask=vocab_mask,
                other=0.0,
            ).to(tl.float32)
            residual = tl.maximum(target - draft, 0.0)

        tl.store(
            block_sums + req_idx * num_vocab_blocks + block_idx,
            tl.sum(residual, axis=0),
        )


@triton.jit
def _chain_rejection_sample_kernel(
    predicts,
    target_probs,
    draft_probs,
    metadata,
    block_sums,
    uniform_samples_for_final_sampling,
    num_draft_tokens: tl.constexpr,
    num_draft_prob_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    vocab_block_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    pad_num_vocab_blocks: tl.constexpr,
):
    req_idx = tl.program_id(0)
    metadata_offset = req_idx * 3
    target_row = tl.load(metadata + metadata_offset).to(tl.int64)
    output_idx = tl.load(metadata + metadata_offset + 1).to(tl.int64)
    all_accepted = tl.load(metadata + metadata_offset + 2).to(tl.int32)

    block_offsets = tl.arange(0, pad_num_vocab_blocks)
    block_mask = block_offsets < num_vocab_blocks
    sums = tl.load(
        block_sums + req_idx * num_vocab_blocks + block_offsets,
        mask=block_mask,
        other=0.0,
    ).to(tl.float32)
    block_cdf = tl.cumsum(sums, axis=0)
    total = tl.sum(sums, axis=0)
    coin = tl.load(uniform_samples_for_final_sampling + req_idx).to(tl.float32)
    target_value = coin * total

    selected_block = tl.sum(
        ((block_cdf <= target_value) & block_mask).to(tl.int32), axis=0
    )
    selected_block = tl.minimum(selected_block, num_vocab_blocks - 1)
    prefix_sum = tl.sum(tl.where(block_offsets < selected_block, sums, 0.0), axis=0)
    local_target = tl.maximum(target_value - prefix_sum, 0.0)

    local_offsets = tl.arange(0, vocab_block_size)
    vocab_offsets = selected_block * vocab_block_size + local_offsets
    vocab_mask = vocab_offsets < vocab_size
    target_offset = (req_idx * num_draft_tokens + target_row) * vocab_size
    target = tl.load(
        target_probs + target_offset + vocab_offsets,
        mask=vocab_mask,
        other=0.0,
    ).to(tl.float32)
    if all_accepted == 1:
        residual = target
    else:
        draft_offset = (req_idx * num_draft_prob_rows + target_row) * vocab_size
        draft = tl.load(
            draft_probs + draft_offset + vocab_offsets,
            mask=vocab_mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.maximum(target - draft, 0.0)

    local_cdf = tl.cumsum(residual, axis=0)
    local_index = tl.sum(
        ((local_cdf <= local_target) & vocab_mask).to(tl.int32), axis=0
    )
    last_valid_local = tl.max(
        tl.where((residual > 0.0) & vocab_mask, local_offsets, -1), axis=0
    )
    valid_local_count = tl.minimum(
        vocab_block_size,
        vocab_size - selected_block * vocab_block_size,
    )
    sampled_local = tl.where(
        local_index < valid_local_count,
        local_index,
        last_valid_local,
    )
    sampled_token = tl.where(
        sampled_local >= 0,
        selected_block * vocab_block_size + sampled_local,
        vocab_size - 1,
    )
    tl.store(predicts + output_idx, tl.minimum(sampled_token, vocab_size - 1))


def chain_speculative_sampling_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """NPU kernel implementation of classic chain rejection sampling."""
    del retrive_next_token, retrive_next_sibling
    del threshold_single, threshold_acc, deterministic

    if candidates.ndim != 2 or target_probs.ndim != 3:
        raise ValueError("candidates must be 2-D and target_probs must be 3-D")
    batch_size, num_draft_tokens = candidates.shape
    if batch_size == 0:
        return
    if num_draft_tokens == 0:
        raise ValueError("num_draft_tokens must be positive")
    if target_probs.shape[:2] != (batch_size, num_draft_tokens):
        raise ValueError(
            "target_probs shape must be [batch, num_draft_tokens, vocab_size]"
        )
    if retrive_index.shape != candidates.shape:
        raise ValueError("retrive_index shape must match candidates")
    if accept_index.shape != candidates.shape:
        raise ValueError("classic rejection sampling requires a topk=1 linear chain")
    if accept_token_num.shape != (batch_size,):
        raise ValueError("accept_token_num must have shape [batch]")
    if predicts.ndim != 1:
        raise ValueError("predicts must be 1-D")
    if uniform_samples.shape != candidates.shape:
        raise ValueError("uniform_samples shape must match candidates")
    if uniform_samples_for_final_sampling.shape != (batch_size,):
        raise ValueError("uniform_samples_for_final_sampling must have shape [batch]")
    if draft_probs is None or draft_probs.ndim != 3:
        raise ValueError("draft_probs must be a 3-D tensor")
    if draft_probs.shape[0] != batch_size:
        raise ValueError("draft_probs batch size must match candidates")
    if draft_probs.shape[1] < max(num_draft_tokens - 1, 1):
        raise ValueError("draft_probs does not contain every proposal row")
    if draft_probs.shape[-1] != target_probs.shape[-1]:
        raise ValueError("draft_probs and target_probs vocab sizes must match")
    if target_probs.dtype != torch.float32 or draft_probs.dtype != torch.float32:
        raise TypeError("target_probs and draft_probs must be torch.float32")
    if uniform_samples.dtype != torch.float32:
        raise TypeError("uniform_samples must be torch.float32")
    if uniform_samples_for_final_sampling.dtype != torch.float32:
        raise TypeError("uniform_samples_for_final_sampling must be torch.float32")
    integer_dtypes = (
        (predicts, torch.int32, "predicts"),
        (accept_index, torch.int32, "accept_index"),
        (accept_token_num, torch.int32, "accept_token_num"),
        (candidates, torch.int64, "candidates"),
        (retrive_index, torch.int64, "retrive_index"),
    )
    for tensor, expected_dtype, name in integer_dtypes:
        if tensor.dtype != expected_dtype:
            raise TypeError(f"{name} must be {expected_dtype}")
    tensors = (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
    )
    if any(tensor.device != target_probs.device for tensor in tensors):
        raise ValueError("all tensors must be on the same NPU device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")

    num_speculative_tokens = accept_index.shape[1]
    num_draft_prob_rows = draft_probs.shape[1]
    vocab_size = target_probs.shape[-1]
    vocab_block_size = 2048
    num_vocab_blocks = triton.cdiv(vocab_size, vocab_block_size)
    pad_num_vocab_blocks = triton.next_power_of_2(num_vocab_blocks)
    _, num_vector_cores = get_device_properties()
    # Cap the launch to physical vector cores; each program strides over blocks.
    num_block_sum_programs = min(batch_size * num_vocab_blocks, num_vector_cores)

    metadata = torch.empty(
        (batch_size, 3), dtype=torch.int64, device=target_probs.device
    )
    block_sums = torch.empty(
        (batch_size, num_vocab_blocks),
        dtype=torch.float32,
        device=target_probs.device,
    )

    _chain_rejection_accept_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        target_probs,
        draft_probs,
        metadata,
        num_draft_tokens=num_draft_tokens,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_prob_rows=num_draft_prob_rows,
        vocab_size=vocab_size,
    )
    _chain_rejection_block_sum_kernel[(num_block_sum_programs,)](
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        num_draft_prob_rows=num_draft_prob_rows,
        vocab_size=vocab_size,
        vocab_block_size=vocab_block_size,
        num_vocab_blocks=num_vocab_blocks,
    )
    _chain_rejection_sample_kernel[(batch_size,)](
        predicts,
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        uniform_samples_for_final_sampling,
        num_draft_tokens=num_draft_tokens,
        num_draft_prob_rows=num_draft_prob_rows,
        vocab_size=vocab_size,
        vocab_block_size=vocab_block_size,
        num_vocab_blocks=num_vocab_blocks,
        pad_num_vocab_blocks=pad_num_vocab_blocks,
    )
