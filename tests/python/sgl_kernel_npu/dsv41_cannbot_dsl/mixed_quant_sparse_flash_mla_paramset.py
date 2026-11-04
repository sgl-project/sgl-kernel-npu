# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE.cannbot-dsl in the root of the software repository for the full text of the License.

"""Local MQSMLA cases. MQSMLA_CASES selects names or all; no saved data required."""

_BASE = dict(
    q_lengths=[1, 1],
    s_ori=257,
    s_cmp=193,
    k_ori=128,
    k_cmp=129,
    block_size_ori=128,
    block_size_cmp=128,
    has_cmp=True,
    kv_axis0_noncontiguous=False,
    return_softmax_lse=True,
)
TEST_PARAMS = {
    "mixed_decode": dict(_BASE),
    "ori_only": dict(_BASE, has_cmp=False, return_softmax_lse=False),
    "empty_and_tail": dict(_BASE, q_lengths=[3, 0, 5], vary_lengths=True),
    # Entire padding subtiles must reuse valid addresses, including when the
    # address table contains only one padded 16-column block.
    "single_token_k": dict(_BASE, q_lengths=[4], k_ori=1, k_cmp=1, vary_lengths=True),
    "narrow_k_tail": dict(_BASE, q_lengths=[4], k_ori=17, k_cmp=65, vary_lengths=True),
    "padded_pages": dict(
        _BASE,
        q_lengths=[2, 3],
        kv_axis0_noncontiguous=True,
        block_size_ori=64,
        block_size_cmp=32,
    ),
    "cmp_only_rows": dict(_BASE, q_lengths=[3], zero_ori=True, vary_lengths=True),
    "omit_ori_length": dict(_BASE, omit_ori_topk_length=True),
    "omit_cmp_length": dict(_BASE, omit_cmp_topk_length=True),
    "omit_both_lengths": dict(
        _BASE, omit_ori_topk_length=True, omit_cmp_topk_length=True
    ),
    "multirow_42": dict(_BASE, q_lengths=[42], vary_lengths=True),
    "multirow_64": dict(_BASE, q_lengths=[64], vary_lengths=True),
    "selective_fd_72": dict(_BASE, q_lengths=[6] * 12, s_cmp=769, k_cmp=512),
}
ENABLED_PARAMS = ["mixed_decode", "ori_only", "empty_and_tail", "padded_pages"]
