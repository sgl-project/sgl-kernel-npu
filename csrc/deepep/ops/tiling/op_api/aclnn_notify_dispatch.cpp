#include <string.h>
#include "graph/types.h"
#include "aclnn_notify_dispatch.h"

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B,
    SOC_VERSION_ASCEND910_93,
    SOC_VERSION_ASCEND910_95,
    SOC_VERSION_ASCEND310P,
    SOC_VERSION_ASCEND310B,
    SOC_VERSION_BS9SX1A,
    SOC_VERSION_ASCEND610Lite,
    SOC_VERSION_ASCEND910_55,
    SOC_VERSION_MC61AM21A
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910_93,SOC_VERSION_ASCEND910B};
uint32_t socSupportListLen = 2;

TensorDesc inputDesc0_0[1] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[1] =
    {{ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[1] =
    {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_3[1] =
    {{ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[1] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[1] =
    {{ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[1] =
    {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_3[1] =
    {{ge::DT_INT32, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 1, outputDesc0_0, 1};
SupportInfo list0_1 = {inputDesc0_1, 1, outputDesc0_1, 1};
SupportInfo list0_2 = {inputDesc0_2, 1, outputDesc0_2, 1};
SupportInfo list0_3 = {inputDesc0_3, 1, outputDesc0_3, 1};
SupportInfo supportInfo0[4] = {list0_0, list0_1, list0_2, list0_3};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 4};

TensorDesc inputDesc1_0[1] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc inputDesc1_1[1] =
    {{ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc inputDesc1_2[1] =
    {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc1_3[1] =
    {{ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc1_0[1] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc1_1[1] =
    {{ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc1_2[1] =
    {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_3[1] =
    {{ge::DT_INT32, ge::FORMAT_ND}};
SupportInfo list1_0 = {inputDesc1_0, 1, outputDesc1_0, 1};
SupportInfo list1_1 = {inputDesc1_1, 1, outputDesc1_1, 1};
SupportInfo list1_2 = {inputDesc1_2, 1, outputDesc1_2, 1};
SupportInfo list1_3 = {inputDesc1_3, 1, outputDesc1_3, 1};
SupportInfo supportInfo1[4] = {list1_0, list1_1, list1_2, list1_3};
OpSocSupportInfo socSupportInfo1= {supportInfo1, 4};

OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
OpSupportList supportList = {opSocSupportList, 2};

[[maybe_unused]] uint32_t NNOPBASE_NotifyDispatch = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIgnoreContinuesInput(void *executor,
                                                   const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIntArrayInput(void *executor, const aclIntArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddBoolArrayInput(void *executor, const aclBoolArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddFloatArrayInput(void *executor, const aclFloatArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicInput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicOutput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index, const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray* array, const size_t index);
extern aclnnStatus NnopbaseAddFloatArrayAttr(void *executor, const aclFloatArray* array, const size_t index);
extern aclnnStatus NnopbaseAddBoolArrayAttr(void *executor, const aclBoolArray* array, const size_t index);
extern aclnnStatus NnopbaseAddArrayAttrWithDtype(void *executor, void *array, const size_t len, const size_t elementSize, const size_t index, const NnopbaseAttrDtype dtype);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace, uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);
extern aclnnStatus NnopbaseAddScalarInput(void *executor, const aclScalar *scalar, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern aclnnStatus NnopbaseAddScalarListInput(void *executor, const aclScalarList *scalarList, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern void NnopbaseAddOpTypeId(void *executor, const uint32_t opTypeId);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name, const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern aclnnStatus NnopbaseSetRef(void *executor, const size_t inputIrIdx, const size_t outputIrIdx);
extern aclnnStatus NnopbaseSetMc2(void *const executor);
extern aclnnStatus NnopbaseSetHcomGroup(void *const executor, char *const group);

#define ACLNN_SUCCESS  0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData,
    int64_t sendCount,
    char *commGroup,
    int64_t rankSize,
    int64_t rankId,
    int64_t localRankSize,
    int64_t localRankId,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnNotifyDispatchTiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "NotifyDispatch";
    char inputDesc[] = {1};
    char outputDesc[] = {1};
    char attrDesc[] = {1, 1, 1, 1, 1, 1};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(sendData);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(out);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, sendData, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&sendCount), sizeof(int64_t), 0, kNnopbaseInt));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(commGroup);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(commGroup), strlen(commGroup) + 1, 1, kNnopbaseString));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&rankSize), sizeof(int64_t), 2, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&rankId), sizeof(int64_t), 3, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&localRankSize), sizeof(int64_t), 4, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&localRankId), sizeof(int64_t), 5, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, out, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetMc2(*executor));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSetHcomGroup(*executor, commGroup));
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "sendData", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "out", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNotifyDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
