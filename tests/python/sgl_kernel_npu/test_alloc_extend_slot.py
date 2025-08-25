import torch
import unittest

page_size_range = None
draft_token_num_range = None

def alloc_extend_pytorch(
    pre_lens, seq_lens, last_loc, free_pages, page_size, extend_lens
):
    # Step 1: compute total extend lengths
    total_extend_tokens = extend_lens.sum().item()

    # Step 2: Build output buffer
    out_indices = torch.empty(total_extend_tokens, dtype=torch.int32, device="cpu")

    part1_ends = torch.min(
        seq_lens, ((pre_lens + page_size - 1) // page_size) * page_size
    )
    part1_lens = (part1_ends - pre_lens).to(device="cpu")

    part2_starts = ((pre_lens + page_size - 1) // page_size) * page_size
    part2_ends = (seq_lens // page_size) * page_size
    part2_lens = (part2_ends - part2_starts).to(device="cpu")

    part3_lens = (seq_lens % page_size).to(device="cpu")

    # Step 3: Allocate and fill
    global_token_idx = 0
    global_page_idx = 0
    global page_size_range
    if page_size_range is None:
        page_size_range = torch.arange(page_size, device="cpu")

    extend_lens = extend_lens.to(device="cpu")
    last_loc = last_loc.to(device="cpu")
    free_pages = free_pages.to(device="cpu")

    for i in range(len(pre_lens)):
        current_token_idx = 0
        if extend_lens[i] == 0:
            continue

        # Part 1: Fill the partial page at the end of pre_len
        if part1_lens[i] > 0:
            indices = last_loc[i] + 1 + torch.arange(part1_lens[i], device="cpu")
            out_indices[global_token_idx : global_token_idx + part1_lens[i]] = indices
            global_token_idx += part1_lens[i]
            current_token_idx += part1_lens[i]
        if (
            global_token_idx >= total_extend_tokens
            or current_token_idx >= extend_lens[i]
        ):
            continue

        # Part 2:Fill full new pages
        if part2_lens[i] > 0:
            num_full_pages = part2_lens[i] // page_size
            page_id = free_pages[global_page_idx : global_page_idx + num_full_pages]
            slots = (page_id[:, None] * page_size + page_size_range).flatten()
            valid_slots = slots[
                : min(len(slots), total_extend_tokens - global_token_idx)
            ]
            out_indices[global_token_idx : global_token_idx + len(valid_slots)] = (
                valid_slots
            )
            global_token_idx += len(valid_slots)
            current_token_idx += len(valid_slots)
            global_page_idx += num_full_pages

        # Part 3: Fill the final partial page
        if (
            part3_lens[i] > 0
            and global_token_idx < total_extend_tokens
            and current_token_idx < extend_lens[i]
        ):
            page = free_pages[global_page_idx]
            slots = page * page_size + torch.arange(part3_lens[i], device="cpu")
            out_indices[global_token_idx : global_token_idx + part3_lens[i]] = slots
            global_token_idx += part3_lens[i]
            current_token_idx += part3_lens[i]
            global_page_idx += 1
    out_indices = out_indices.to(device="npu")
    return out_indices, total_extend_tokens


def alloc_extend_native(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    estimated_num_new_pages,
    speculative_num_draft_tokens=None,
):
    extend_lens = seq_lens - prefix_lens
    indices_1 = torch.where(extend_lens == 1)[0]  # no accept
    indices_2 = torch.where(extend_lens == 2)[0]  # accept one token
    if len(indices_1) > 0 and len(indices_2) == 0:
        out_indices = last_loc + 1
        if estimated_num_new_pages > 0:
            need_new_page_idx = torch.where(seq_lens % page_size == 1)[0]
            out_indices = out_indices.to(dtype=torch.int32)
            out_indices[need_new_page_idx] = (
                page_size * free_pages[: len(need_new_page_idx)]
            )
        out_indices = out_indices.to(dtype=torch.int32)
    elif (
        len(indices_1) == 0 and len(indices_2) > 0 and speculative_num_draft_tokens == 2
    ):
        global draft_token_num_range
        if draft_token_num_range is None:
            draft_token_num_range = torch.arange(
                1, speculative_num_draft_tokens + 1, device=prefix_lens.device
            )
        out_indices = (
            last_loc.repeat_interleave(speculative_num_draft_tokens).reshape(
                -1, speculative_num_draft_tokens
            )
            + draft_token_num_range
        )
        out_indices = out_indices.to(dtype=torch.int32)
        if estimated_num_new_pages > 0:
            need_new_page_idx_1 = torch.where((prefix_lens + 1) % page_size == 0)[0]
            need_new_page_idx_2 = torch.where(prefix_lens % page_size == 0)[0]
            sorted_need_new_page_idx, _ = torch.sort(
                torch.cat((need_new_page_idx_1, need_new_page_idx_2))
            )
            new_pages_first_token_idx = page_size * free_pages[:estimated_num_new_pages]
            new_page_idx_1_positions = torch.searchsorted(
                sorted_need_new_page_idx, need_new_page_idx_1
            )
            new_page_idx_2_positions = torch.searchsorted(
                sorted_need_new_page_idx, need_new_page_idx_2
            )
            out_indices[need_new_page_idx_1, 1] = new_pages_first_token_idx[
                new_page_idx_1_positions
            ]
            out_indices[need_new_page_idx_2, 0] = new_pages_first_token_idx[
                new_page_idx_2_positions
            ]
            out_indices[need_new_page_idx_2, 1] = (
                out_indices[need_new_page_idx_2, 0] + 1
            )
        out_indices = out_indices.view(-1).to(dtype=torch.int32)
    else:
        out_indices, estimated_num_new_pages = alloc_extend_pytorch(
            prefix_lens, seq_lens, last_loc, free_pages, page_size, extend_lens
        )
        out_indices = out_indices.to(dtype=torch.int32)

    return out_indices, estimated_num_new_pages


class TestAllocExtend(unittest.TestCase):
    def compute(
        self,
        prefix_lens, 
        seq_lens, 
        last_loc, 
        free_pages, 
        page_size, 
        estimated_num_new_pages,
        dtype,
        device,
    ):
        out_indices_gt, estimated_num_new_pages_gt = alloc_extend_native(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages
        )
        print("=================")
        print(f"gt out_indices: {out_indices_gt}, estimated_num_new_pages: {estimated_num_new_pages_gt}")
        print("=================")
        import sgl_kernel_npu
        extend_lens = seq_lens - prefix_lens
        extend_tokens_num = extend_lens.sum().item()
        out_indices = torch.empty((extend_tokens_num,), dtype=dtype, device=device)
        estimated_num_new_pages = torch.empty((1,), dtype=dtype, device=device)
        torch.ops.npu.alloc_extend(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages,
            page_size,
            out_indices,
            estimated_num_new_pages,
        )
        print("=================")
        merged_value = estimated_num_new_pages.item()
        num_new_pages = merged_value >> 32
        print(f"sgl-kernel out_indices: {out_indices}, estimated_num_new_pages: {estimated_num_new_pages}, num_new_pages: {num_new_pages}")
        print("=================")
        ret = torch.equal(out_indices_gt, out_indices)
        self.assertTrue(ret)
        # self.assertEqual(estimated_num_new_pages_gt, num_new_pages)

    def test_case1_prefill(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([0], dtype=dtype, device=device)
        seq_lens = torch.tensor([7], dtype=dtype, device=device)
        last_loc = torch.tensor([-1], dtype=dtype, device=device)
        free_pages = torch.arange(1, 100, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )

    def test_case2_decoder(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([7], dtype=dtype, device=device)
        seq_lens = torch.tensor([8], dtype=dtype, device=device)
        last_loc = torch.tensor([134], dtype=dtype, device=device)
        free_pages = torch.arange(1, 1000, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )

    def test_case3_decoder_verfiy(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([7], dtype=dtype, device=device)
        seq_lens = torch.tensor([9], dtype=dtype, device=device)
        last_loc = torch.tensor([134], dtype=dtype, device=device)
        free_pages = torch.arange(1, 1000, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )
    
    def test_case4_prefill_multi_pages(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([0], dtype=dtype, device=device)
        seq_lens = torch.tensor([300], dtype=dtype, device=device)
        last_loc = torch.tensor([-1], dtype=dtype, device=device)
        free_pages = torch.arange(1, 1000, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )
    
    def test_case5_prefill_cache_multi_pages(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([100], dtype=dtype, device=device)
        seq_lens = torch.tensor([400], dtype=dtype, device=device)
        last_loc = torch.tensor([227], dtype=dtype, device=device)
        free_pages = torch.arange(1, 1000, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )
    
    def test_case6_prefill_multi_batch(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([0] * 19, dtype=dtype, device=device)
        seq_lens = torch.tensor([734, 720, 705, 778, 725, 710, 772, 735, 726, 729, 732, 737, 720, 728, 720, 761, 723, 735, 698], dtype=dtype, device=device)
        last_loc = torch.tensor([-1] * 19, dtype=dtype, device=device)
        free_pages = torch.arange(1, 1066, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )
        
    def test_case7_decoder_multi_batch(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([697, 734, 720, 705, 778, 725, 710, 772, 735, 726, 729, 732, 737, 720, 728, 720, 761, 723, 735, 698], dtype=dtype, device=device)
        seq_lens = torch.tensor([698, 735, 721, 706, 779, 726, 711, 773, 736, 727, 730, 733, 738, 721, 729, 721, 762, 724, 736, 699], dtype=dtype, device=device)
        last_loc = torch.tensor([1720, 2525, 3279, 4032, 4873, 5716, 6469, 7299, 8158, 8917, 9688, 10459, 11232, 11983, 12759, 13519, 14328, 15058, 15838, 16569], dtype=dtype, device=device)
        free_pages = torch.arange(1, 950, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )

    def test_case8_decoder_multi_batch_verfiy(self):
        dtype = torch.int64
        device = 'npu'
        prefix_lens = torch.tensor([697, 734, 720, 705, 778, 725, 710, 772, 735, 726, 729, 732, 737, 720, 728, 720, 761, 723, 735, 698], dtype=dtype, device=device)
        seq_lens = torch.tensor([699, 736, 722, 707, 780, 727, 712, 774, 737, 728, 731, 734, 739, 722, 730, 722, 763, 725, 737, 700], dtype=dtype, device=device)
        last_loc = torch.tensor([1720, 2525, 3279, 4032, 4873, 5716, 6469, 7299, 8158, 8917, 9688, 10459, 11232, 11983, 12759, 13519, 14328, 15058, 15838, 16569], dtype=dtype, device=device)
        free_pages = torch.arange(1, 950, dtype=dtype, device=device)
        page_size = 128
        estimated_num_new_pages = (
            (
                (seq_lens + page_size - 1) // page_size
                - (prefix_lens + page_size - 1) // page_size
            )
            .sum()
            .item()
        )
        self.compute(
            prefix_lens, 
            seq_lens, 
            last_loc, 
            free_pages, 
            page_size, 
            estimated_num_new_pages,
            dtype,
            device,
        )


if __name__ == "__main__":
    unittest.main()