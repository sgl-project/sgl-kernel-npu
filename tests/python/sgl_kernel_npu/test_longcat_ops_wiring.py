from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]


def read_repo_file(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class TestLongcatOpsWiring(unittest.TestCase):
    def test_compute_n_gram_ids_wiring_is_complete(self):
        header = read_repo_file("include/sgl_kenel_npu_ops.h")
        registry = read_repo_file("csrc/pytorch_extensions.cpp")
        cmake = read_repo_file("csrc/CMakeLists.txt")
        host = read_repo_file("csrc/compute_n_gram_ids/op_host/compute_n_gram_ids.cpp")

        self.assertIn("at::Tensor compute_n_gram_ids(", header)
        self.assertIn('m.def(\n        "compute_n_gram_ids(', registry)
        self.assertIn('m.impl("compute_n_gram_ids", TORCH_FN(sglang::npu_kernel::compute_n_gram_ids));', registry)
        self.assertIn("${PROJECT_OP_SRC_BASE}/compute_n_gram_ids/op_host/compute_n_gram_ids.cpp", cmake)
        self.assertIn("${PROJECT_OP_SRC_BASE}/compute_n_gram_ids/op_kernel/compute_n_gram_ids.cpp", cmake)
        self.assertIn("uint32_t total_task = static_cast<uint32_t>(batch_size * (oe_n - 1) * oe_k);", host)
        self.assertIn("block_dim = std::min(aiv_num, std::max(total_task, 1U));", host)
        self.assertIn("auto output = at::empty({tokens.size(0), (oe_n - 1) * oe_k},", host)
        self.assertIn("constexpr uint32_t MAX_CAPTURE_NUM = 1024U;", host)
        self.assertIn("static std::unordered_map<uint64_t, uint32_t> captureMap;", host)
        self.assertIn("static at::Tensor globalTilingBuffer;", host)
        self.assertIn("host_utils::TupleHasher::Hash", host)

    def test_mlp_lightning_indexer_contract_and_optional_launch(self):
        header = read_repo_file("include/sgl_kenel_npu_ops.h")
        registry = read_repo_file("csrc/pytorch_extensions.cpp")
        wrapper = read_repo_file("csrc/mlp_lightning_indexer/op_host/mlp_lightning_indexer.cpp")
        tiling = read_repo_file("csrc/mlp_lightning_indexer/op_host/tiling/mlp_lightning_indexer_tiling.cpp")
        helper = read_repo_file("csrc/utils/torch_helper.h")

        self.assertIn("std::tuple<at::Tensor, at::Tensor> mlp_lightning_indexer(", header)
        self.assertIn('"mlp_lightning_indexer(Tensor query, Tensor key, Tensor weights, "', registry)
        self.assertIn("-> (Tensor, Tensor)", registry)
        self.assertIn("weights must be float32", tiling)
        self.assertIn("cur_seq_lengths_key only supports int64", tiling)
        self.assertIn("cur_seq_lengths_query only supports int64", tiling)
        self.assertIn("size = static_cast<uint32_t>(tensor->GetShapeSize()) - 1;", tiling)
        self.assertIn("liInfo.returnValue = static_cast<int8_t>(*attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX));", tiling)
        self.assertIn("ConvertType(const c10::optional<at::Tensor> &at_tensor)", helper)
        self.assertIn("return nullptr;", helper)
        self.assertIn("uint32_t blockDim = tilingData.usedCoreNum;", wrapper)
        self.assertIn("EXEC_KERNEL_CMD(mlp_lightning_indexer, blockDim, query, key, weights, cur_seq_lengths_query,", wrapper)
        self.assertIn("cur_seq_lengths_key, block_table, init_tensor, local_tensor, sparse_indices, sparse_values,", wrapper)
        self.assertIn("constexpr uint32_t MAX_CAPTURE_NUM = 1024U;", wrapper)
        self.assertIn("static std::unordered_map<uint64_t, uint32_t> captureMap;", wrapper)
        self.assertIn("static at::Tensor globalTilingBuffer;", wrapper)
        self.assertIn("GetOrCreateCachedTilingTensor", wrapper)
        self.assertIn("host_utils::TupleHasher::Hash", wrapper)

    def test_update_oe_token_table_wiring_is_complete(self):
        header = read_repo_file("include/sgl_kenel_npu_ops.h")
        registry = read_repo_file("csrc/pytorch_extensions.cpp")
        cmake = read_repo_file("csrc/CMakeLists.txt")
        host = read_repo_file("csrc/update_oe_token_table/op_host/update_oe_token_table.cpp")
        kernel = read_repo_file("csrc/update_oe_token_table/op_kernel/update_oe_token_table.h")

        self.assertIn("at::Tensor update_oe_token_table(", header)
        self.assertIn('"update_oe_token_table(Tensor tokens, Tensor req_lens, Tensor row_indices, Tensor column_starts, "', registry)
        self.assertIn('m.impl("update_oe_token_table", TORCH_FN(sglang::npu_kernel::update_oe_token_table));', registry)
        self.assertIn("${PROJECT_OP_SRC_BASE}/update_oe_token_table/op_host/update_oe_token_table.cpp", cmake)
        self.assertIn("${PROJECT_OP_SRC_BASE}/update_oe_token_table/op_kernel/update_oe_token_table.cpp", cmake)
        self.assertIn("int64_t ignore_ub_size = CeilAlign(ignore_token_num * static_cast<int64_t>(sizeof(int32_t)), BLOCK_SIZE);", host)
        self.assertIn("int32_t ub_factor =", host)
        self.assertIn("EXEC_KERNEL_CMD(update_oe_token_table, block_dim, tokens, req_lens, row_indices, column_starts,", host)
        self.assertIn("Duplicate(minusTensor, static_cast<int32_t>(-1), count);", kernel)
        self.assertIn("auto alignCnt = CeilDiv(count, BLOCK_SIZE) * BLOCK_SIZE;", kernel)
        self.assertIn("constexpr uint32_t MAX_CAPTURE_NUM = 1024U;", host)
        self.assertIn("static std::unordered_map<uint64_t, uint32_t> captureMap;", host)
        self.assertIn("static at::Tensor globalTilingBuffer;", host)
        self.assertIn("host_utils::TupleHasher::Hash", host)


if __name__ == "__main__":
    unittest.main()
