/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_A2_
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_A2_

#include <string>
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现MoeDistributeCombine功能。
 * @brief aclnnMoeDistributeCombine的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] expandX: 计算输入，Tensor，数据类型float16，bfloat16，必须为2维，数据格式支持ND。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] expandIdx: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND，token由多少个MOE专家处理。
 * @param [in] epSendCounts: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] expertScales: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。
 * @param [in] tpSendCounts: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。无tp域通信时传空。
 * @param [in] xActiveMask: 计算输入，Tensor，数据类型bool，必须为1维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] activationScale: 计算输入，Tensor，数据类型float32，必须为1维，数据格式支持ND, GMM外抛的左矩阵量化系数，
    当x的类型为int32时，该参数必须有。预留参数，暂未使用，传空即可。
 * @param [in] weightScale: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND, GMM外抛的右矩阵量化系数，
    当x的类型为int32时，该参数必须有。预留参数，暂未使用，传空即可。
 * @param [in] groupList: 计算输入，Tensor，数据类型int64，必须为1维，数据格式支持ND, GMM的分组大小，
    当weight scale的E>1时，该参数必须有。预留参数，暂未使用，传空即可。
 * @param [in] expandScales:
 计算输入，Tensor，数据类型float32，必须为1维，数据格式支持ND。在昇腾910_93中暂未使用，传空即可。
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。字符串长度范围为[1, 128)，不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。在昇腾910_93场景中取值支持8/16/32/64/128/144/256/288。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。在昇腾910_93场景中取值范围[1, 512]，且需
    满足moeExpertNum%(epWorldSize-sharedExpertRankNum)等于0。
 * @param [in] groupTp: 计算可选输入，str。tp通信域名称，数据并行的通信域。无tp通信域时传空，
    有tp通信域时字符串长度范围为[1, 128)，不能和groupEp相同。
 * @param [in] tpWorldSize: 计算可选输入，int。tp通信域size。取值范围[0, 2]，0和1表示无tp域通信，有tp域通信时仅支持2。
 * @param [in] tpRankId: 计算可选输入，int。tp本卡Id。取值范围[0, tpWorldSize)，同一个TP通信域中各卡的tpRankId不能重复。
 * @param [in] expertShardType: 计算可选输入，int。专家共享类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。取值范围[0, 1]。0表示无共享专家。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家数量。支持传0表示无共享专家卡，不为0时需满足
    sharedExpertRankNum<epWorldSize且epWorldSize%sharedExpertRankNum等于0。
 * @param [in] globalBs: 计算可选输入，int。ep域全局的batch size大小。支持传0表示使用默认值。
 * @param [in] outDtype: 计算可选输入，int。输出数据类型。预留参数，暂未使用，传0即可。
 * @param [in] commQuantMode: 计算可选输入，int。通信量化类型。预留参数，暂未使用，传0即可。
 * @param [in] groupListType: 计算可选输入，int。groupList格式。预留参数，暂未使用，传0即可。
 * @param [out] x: 计算输出，Tensor，必选输出，数据类型支持float16, bfloat16，仅支持2维，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeDistributeCombineA2GetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *expandIdx, const aclTensor *epSendCounts,
    const aclTensor *expertScales, const aclTensor *tpSendCounts, const aclTensor *xActiveMask,
    const aclTensor *activationScale, const aclTensor *weightScale, const aclTensor *groupList,
    const aclTensor *expandScales, const aclTensor *offsetInner, const aclTensor *offsetOuter,
    const aclTensor *countOuter, char *groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMoeDistributeCombine的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeDistributeCombineA2(void *workspace, uint64_t workspaceSize,
                                                                               aclOpExecutor *executor,
                                                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_COMBINE_
