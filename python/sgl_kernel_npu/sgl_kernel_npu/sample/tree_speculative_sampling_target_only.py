import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _tree_target_only_accept_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    target_probs,
    rejected_probs,
    metadata,
    threshold_single,
    threshold_acc,
    num_draft_tokens: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    vocab_size: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_offset = req_idx * num_draft_tokens

    cur_prob_row = tl.full((), 0, tl.int64)
    cur_node = tl.full((), 0, tl.int64)
    last_accepted_idx = tl.load(retrive_index + row_offset).to(tl.int64)
    coin = tl.load(uniform_samples + row_offset).to(tl.float32)
    num_accepted = 0
    path_active = tl.full((), 1, tl.int32)

    tl.store(accept_index + req_idx * num_speculative_tokens, last_accepted_idx)

    # This is the same breadth-at-each-depth traversal used by the CUDA kernel:
    # descend to the first child, then walk siblings until one is accepted.
    for _depth in range(1, num_speculative_tokens):
        accepted_at_depth = tl.full((), 0, tl.int32)
        prob_acc = tl.full((), 0.0, tl.float32)

        if path_active == 1:
            cur_node = tl.load(retrive_next_token + row_offset + cur_node).to(tl.int64)
            if cur_node == -1:
                path_active = 0

        # The loop is bounded by the number of tree nodes. It terminates
        # logically when a child is accepted or the sibling list reaches -1.
        for _sibling in range(0, num_draft_tokens):
            if (path_active == 1) & (accepted_at_depth == 0) & (cur_node != -1):
                draft_token = tl.load(candidates + row_offset + cur_node).to(tl.int64)
                draft_idx = tl.load(retrive_index + row_offset + cur_node).to(tl.int64)
                prob_offset = (row_offset + cur_prob_row) * vocab_size + draft_token
                target_prob_single = tl.load(target_probs + prob_offset).to(tl.float32)
                prob_acc += target_prob_single

                accepted = (coin <= prob_acc / threshold_acc) | (
                    target_prob_single >= threshold_single
                )
                if accepted:
                    tl.store(predicts + last_accepted_idx, draft_token)
                    num_accepted += 1
                    tl.store(
                        accept_index + req_idx * num_speculative_tokens + num_accepted,
                        draft_idx,
                    )
                    last_accepted_idx = draft_idx
                    cur_prob_row = cur_node
                    coin = tl.load(uniform_samples + row_offset + cur_node).to(
                        tl.float32
                    )
                    accepted_at_depth = 1
                else:
                    # The CUDA target-only kernel stores the rejected sibling's
                    # target probability in draft_probs and later samples from
                    # relu(target_probs - draft_probs).
                    tl.store(rejected_probs + prob_offset, target_prob_single)
                    cur_node = tl.load(retrive_next_sibling + row_offset + cur_node).to(
                        tl.int64
                    )

        if accepted_at_depth == 0:
            path_active = 0

    tl.store(accept_token_num + req_idx, num_accepted)

    # metadata = [final target-probability row, final output slot].
    metadata_offset = req_idx * 2
    tl.store(metadata + metadata_offset, cur_prob_row)
    tl.store(metadata + metadata_offset + 1, last_accepted_idx)


@triton.jit
def _tree_target_only_block_sum_kernel(
    target_probs,
    rejected_probs,
    metadata,
    block_sums,
    batch_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
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

        target_row = tl.load(metadata + req_idx * 2).to(tl.int64)
        probs_offset = (req_idx * num_draft_tokens + target_row) * vocab_size
        target = tl.load(
            target_probs + probs_offset + vocab_offsets,
            mask=vocab_mask,
            other=0.0,
        ).to(tl.float32)
        rejected = tl.load(
            rejected_probs + probs_offset + vocab_offsets,
            mask=vocab_mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.maximum(target - rejected, 0.0)
        block_sum = tl.sum(residual, axis=0)
        tl.store(block_sums + req_idx * num_vocab_blocks + block_idx, block_sum)


@triton.jit
def _tree_target_only_sample_kernel(
    predicts,
    target_probs,
    rejected_probs,
    metadata,
    block_sums,
    uniform_samples_for_final_sampling,
    num_draft_tokens: tl.constexpr,
    vocab_size: tl.constexpr,
    vocab_block_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    pad_num_vocab_blocks: tl.constexpr,
):
    req_idx = tl.program_id(0)
    metadata_offset = req_idx * 2
    target_row = tl.load(metadata + metadata_offset).to(tl.int64)
    output_idx = tl.load(metadata + metadata_offset + 1).to(tl.int64)

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
    target = coin * total

    selected_block = tl.sum(((block_cdf <= target) & block_mask).to(tl.int32), axis=0)
    selected_block = tl.minimum(selected_block, num_vocab_blocks - 1)
    prefix_sum = tl.sum(tl.where(block_offsets < selected_block, sums, 0.0), axis=0)
    local_target = tl.maximum(target - prefix_sum, 0.0)

    local_offsets = tl.arange(0, vocab_block_size)
    vocab_offsets = selected_block * vocab_block_size + local_offsets
    vocab_mask = vocab_offsets < vocab_size
    probs_offset = (req_idx * num_draft_tokens + target_row) * vocab_size
    target_probs_block = tl.load(
        target_probs + probs_offset + vocab_offsets,
        mask=vocab_mask,
        other=0.0,
    ).to(tl.float32)
    rejected_probs_block = tl.load(
        rejected_probs + probs_offset + vocab_offsets,
        mask=vocab_mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.maximum(target_probs_block - rejected_probs_block, 0.0)

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
    sampled_token = tl.minimum(sampled_token, vocab_size - 1)
    tl.store(predicts + output_idx, sampled_token)


def tree_speculative_sampling_target_only(
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
    """NPU port of GPU target-only tree speculative sampling.

    ``draft_probs`` is scratch storage, matching the GPU API. The function
    clears it and records rejected sibling probabilities before sampling from
    ``relu(target_probs - draft_probs)`` on the final selected tree row.
    """
    del deterministic

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
    tree_shapes = (
        retrive_index.shape,
        retrive_next_token.shape,
        retrive_next_sibling.shape,
        uniform_samples.shape,
    )
    if any(shape != candidates.shape for shape in tree_shapes):
        raise ValueError("all tree-index and uniform tensors must match candidates")
    if accept_index.ndim != 2 or accept_index.shape[0] != batch_size:
        raise ValueError("accept_index must be [batch, max_tree_depth]")
    num_speculative_tokens = accept_index.shape[1]
    if not 1 <= num_speculative_tokens <= num_draft_tokens:
        raise ValueError("max_tree_depth must be in [1, num_draft_tokens]")
    if accept_token_num.shape != (batch_size,):
        raise ValueError("accept_token_num must have shape [batch]")
    if predicts.ndim != 1:
        raise ValueError("predicts must be 1-D")
    if uniform_samples_for_final_sampling.shape != (batch_size,):
        raise ValueError("uniform_samples_for_final_sampling must have shape [batch]")
    if draft_probs.shape != target_probs.shape:
        raise ValueError("draft_probs scratch must match target_probs")
    if draft_probs.data_ptr() == target_probs.data_ptr():
        raise ValueError("draft_probs must not alias target_probs")
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
        (retrive_next_token, torch.int64, "retrive_next_token"),
        (retrive_next_sibling, torch.int64, "retrive_next_sibling"),
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
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
    )
    if any(tensor.device != target_probs.device for tensor in tensors):
        raise ValueError("all tensors must be on the same NPU device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")
    if not 0.0 <= threshold_single <= 1.0:
        raise ValueError("threshold_single must be in [0, 1]")
    if not 0.0 <= threshold_acc <= 1.0:
        raise ValueError("threshold_acc must be in [0, 1]")

    threshold_acc = max(float(threshold_acc), 1e-9)
    vocab_size = target_probs.shape[-1]
    vocab_block_size = 2048
    num_vocab_blocks = triton.cdiv(vocab_size, vocab_block_size)
    pad_num_vocab_blocks = triton.next_power_of_2(num_vocab_blocks)
    _, num_vector_cores = get_device_properties()
    # Cap the launch to physical vector cores; each program strides over blocks.
    num_block_sum_programs = min(batch_size * num_vocab_blocks, num_vector_cores)

    # The CUDA call site passes zeros_like(target_probs). Clearing in the NPU
    # wrapper makes the scratch contract explicit and permits empty_like callers.
    draft_probs.zero_()
    metadata = torch.empty(
        (batch_size, 2), dtype=torch.int64, device=target_probs.device
    )
    block_sums = torch.empty(
        (batch_size, num_vocab_blocks),
        dtype=torch.float32,
        device=target_probs.device,
    )

    _tree_target_only_accept_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        target_probs,
        draft_probs,
        metadata,
        float(threshold_single),
        threshold_acc,
        num_draft_tokens=num_draft_tokens,
        num_speculative_tokens=num_speculative_tokens,
        vocab_size=vocab_size,
    )
    _tree_target_only_block_sum_kernel[(num_block_sum_programs,)](
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        vocab_block_size=vocab_block_size,
        num_vocab_blocks=num_vocab_blocks,
    )
    _tree_target_only_sample_kernel[(batch_size,)](
        predicts,
        target_probs,
        draft_probs,
        metadata,
        block_sums,
        uniform_samples_for_final_sampling,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        vocab_block_size=vocab_block_size,
        num_vocab_blocks=num_vocab_blocks,
        pad_num_vocab_blocks=pad_num_vocab_blocks,
    )
