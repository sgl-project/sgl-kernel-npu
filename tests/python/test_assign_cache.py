import torch
import torch_npu
import sgl_kernel_npu
import unittest
import time

class TestAssignCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        op = torch.ops.npu.assign_cache_op
        print("Type:", type(op))
        print("Repr:", op)
        print("Overload names:", op.overloads())
        default_overload = getattr(op, "default", None)
        if default_overload:
            print("Schema:", default_overload._schema)

        cls.batch_size = 96
        cls.req_pool_indices = torch.arange(0, cls.batch_size // 2, device='npu', dtype=torch.int32)
        cls.steps = 32

    @classmethod
    def assign_req_to_token_pool_native(
        cls,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
        bs: int,
    ):
        out_cache_loc_length = end_offset - start_offset
        token_pool = req_to_token[req_pool_indices]
        out_cache_loc_cumsum_length = torch.cumsum(out_cache_loc_length, dim=0)
        out_cache_loc_start_idx = torch.cat((torch.tensor([0], device=req_to_token.device), out_cache_loc_cumsum_length))

        for i in range(bs):
            token_pool[i][start_offset[i]:end_offset[i]] = out_cache_loc[
                                                        out_cache_loc_start_idx[i]:out_cache_loc_cumsum_length[i]]
        req_to_token[req_pool_indices] = token_pool

    @classmethod
    def assign_req_to_token_pool_ascendc(
        cls,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
        bs: int,
    ):
        out_cache_loc_length = end_offset - start_offset
        token_pool = req_to_token[req_pool_indices]
        out_cache_loc_cumsum_length = torch.cumsum(out_cache_loc_length, dim=0)
        out_cache_loc_start_idx = torch.cat((torch.tensor([0], device=req_to_token.device),
            out_cache_loc_cumsum_length))
        torch.ops.npu.assign_cache_op(token_pool, out_cache_loc, start_offset, end_offset,
            out_cache_loc_start_idx, out_cache_loc_cumsum_length)
        req_to_token[req_pool_indices] = token_pool

    def test_support_all_dtype(self):
        # [50000, 100000] is large data case, it should cause tiling
        token_gaps = [1, 2, 50000]
        seq_lens = [1024, 1024, 100000]
        dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
        for token_gap, seq_len in zip(token_gaps, seq_lens):
            for test_dtype in dtypes:
                start_offset = torch.randint(low=0, high=seq_len - token_gap - 1, size=(self.batch_size,), device="npu")
                end_offset = start_offset + token_gap
                req_to_token = torch.randint(32, (self.batch_size, seq_len), dtype=test_dtype, device='npu')
                req_to_token_dup = req_to_token.clone()
                out_cache_loc = torch.randint(32, (self.batch_size * token_gap,), dtype=test_dtype, device='npu')
                for i in range(self.steps):
                    torch.npu.synchronize()
                    self.assign_req_to_token_pool_native(self.req_pool_indices, req_to_token, start_offset,
                        end_offset, out_cache_loc, self.batch_size // 2)
                    torch.npu.synchronize()
                    self.assign_req_to_token_pool_ascendc(self.req_pool_indices, req_to_token_dup, start_offset,
                        end_offset, out_cache_loc, self.batch_size // 2)
                    torch.npu.synchronize()
                    self.assertEqual(torch.sum(req_to_token).item(), torch.sum(req_to_token_dup).item())
                    self.assertTrue(torch.equal(req_to_token, req_to_token_dup))

    def test_performance(self):
        token_gaps = [1, 2, 50000]
        seq_lens = [1024, 1024, 100000]
        for token_gap, seq_len in zip(token_gaps, seq_lens):
            time1 = 0
            time2 = 0
            start_offset = torch.randint(low=0, high=seq_len - token_gap - 1, size=(self.batch_size,), device="npu", dtype=torch.int32)
            end_offset = start_offset + token_gap
            req_to_token = torch.randint(32, (self.batch_size, seq_len), dtype=torch.int32, device='npu')
            req_to_token_dup = req_to_token.clone()
            out_cache_loc = torch.randint(32, (self.batch_size * token_gap,), dtype=torch.int32, device='npu')
            for i in range(self.steps):
                start = time.time()
                self.assign_req_to_token_pool_native(self.req_pool_indices, req_to_token, start_offset,
                    end_offset, out_cache_loc, self.batch_size // 2)
                torch.npu.synchronize()
                time1 += time.time() - start
                start = time.time()
                self.assign_req_to_token_pool_ascendc(self.req_pool_indices, req_to_token_dup, start_offset,
                    end_offset, out_cache_loc, self.batch_size // 2)
                torch.npu.synchronize()
                time2 += time.time() - start
            self.assertGreater(time1, time2)
            print(f"\nnative time is {time1 / self.steps}, ascendc time is {time2 / self.steps}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
