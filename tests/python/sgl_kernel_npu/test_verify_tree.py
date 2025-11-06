import pytest
import torch
from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy
from sgl_kernel_npu.speculative import verify_tree_greedy_native


def test_verify_tree_greedy():
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device="npu",
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device="npu",
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device="npu",
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device="npu",
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device="npu")
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i][j]) < 10:
                target_logits[i][j][18] = 10

    target_predict = torch.argmax(target_logits, dim=-1)
    predict_shape = (12,)

    bs = candidates.shape[0]
    num_spec_step = 4

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device="npu")  # mutable
    accept_index = torch.full(
        (bs, num_spec_step), -1, dtype=torch.int32, device="npu"
    )  # mutable
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device="npu")  # mutable

    predicts, accept_index, accept_token_num = verify_tree_greedy_native(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        topk=4,
    )
    # Check the expected output.
    assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index.tolist() == [
        [0, 3, 4, 5],
        [6, 10, 11, -1],
    ]
    assert accept_token_num.tolist() == [3, 2]


def test_verify_tree_greedy_simple():
    candidates = torch.tensor(
        [[10375, 28], [223, 15098]], device="npu:0", dtype=torch.int32
    )
    target_predict = torch.tensor(
        [[28, 334], [15098, 18]], device="npu:0", dtype=torch.int32
    )
    retrive_next_sibling = torch.tensor(
        [[-1, -1], [-1, -1]], device="npu:0", dtype=torch.int32
    )
    retrive_index = torch.tensor([[0, 1], [2, 3]], device="npu:0", dtype=torch.int32)
    retrive_next_token = torch.tensor(
        [[1, -1], [1, -1]], device="npu:0", dtype=torch.int32
    )
    predicts = torch.zeros(4, device="npu:0", dtype=torch.int32)
    accept_index = torch.full((2, 2), -1, device="npu:0", dtype=torch.int32)
    accept_token_num = torch.zeros(2, device="npu:0", dtype=torch.int32)

    verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )

    predicts_gt = torch.zeros_like(predicts)
    accept_index_gt = torch.zeros_like(accept_index) * -1
    accept_token_num_gt = torch.zeros_like(accept_token_num)
    predicts_gt, accept_index_gt, accept_token_num_gt = verify_tree_greedy_native(
        predicts_gt,
        accept_index_gt,
        accept_token_num_gt,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        1,
    )
    assert torch.allclose(predicts_gt, predicts)
    assert torch.allclose(accept_index_gt, accept_index)
    assert torch.allclose(accept_token_num_gt, accept_token_num)


if __name__ == "__main__":
    pytest.main([__file__])
