/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ERROR_MANAGER_H_
#define ERROR_MANAGER_H_

#include <cinttypes>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdlib>

namespace error_message {
using char_t = char;
std::string TrimPath(const std::string &str);
#ifdef __GNUC__
int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...)
    __attribute__((format(printf, 3, 4)));

void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...) __attribute__((format(printf, 5, 6)));
#else
int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...);
void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...);
#endif
} // namespace error_message

constexpr size_t const LIMIT_PER_MESSAGE = 1024U;

///
/// @brief Report error message
/// @param [in] key: vector parameter key
/// @param [in] value: vector parameter value
///
#define REPORT_INPUT_ERROR(error_code, key, value) \
    ErrorManager::GetInstance().ATCReportErrMessage(error_code, key, value)

///
/// @brief Report error message
/// @param [in] key: vector parameter key
/// @param [in] value: vector parameter value
///
#define REPORT_ENV_ERROR(error_code, key, value) ErrorManager::GetInstance().ATCReportErrMessage(error_code, key, value)

#define REPORT_INNER_ERROR(error_code, fmt, ...) \
    error_message::ReportInnerError(__FILE__, &__FUNCTION__[0], __LINE__, (error_code), (fmt), ##__VA_ARGS__)

#define REPORT_CALL_ERROR REPORT_INNER_ERROR

namespace error_message {
// first stage
extern const std::string kInitialize;
extern const std::string kModelCompile;
extern const std::string kModelLoad;
extern const std::string kModelExecute;
extern const std::string kFinalize;

// SecondStage
// INITIALIZE
extern const std::string kParser;
extern const std::string kOpsProtoInit;
extern const std::string kSystemInit;
extern const std::string kEngineInit;
extern const std::string kOpsKernelInit;
extern const std::string kOpsKernelBuilderInit;
// MODEL_COMPILE
extern const std::string kPrepareOptimize;
extern const std::string kOriginOptimize;
extern const std::string kSubGraphOptimize;
extern const std::string kMergeGraphOptimize;
extern const std::string kPreBuild;
extern const std::string kStreamAlloc;
extern const std::string kMemoryAlloc;
extern const std::string kTaskGenerate;
// COMMON
extern const std::string kOther;

struct Context {
    uint64_t work_stream_id;
    std::string first_stage;
    std::string second_stage;
    std::string log_header;
};

enum class ErrorMsgMode : uint32_t {
    // 0:内置模式，推理采用线程粒度，训练采用session粒度，1：以进程为粒度
    INTERNAL_MODE    = 0U,
    PROCESS_MODE     = 1U,
    ERR_MSG_MODE_MAX = 2U
};

struct ErrorItem {
    std::string error_id;
    std::string error_message;
    std::string possible_cause;
    std::string solution;
    // args_map作用于有error_id对应的json文件配置时，填充error_message对象
    std::map<std::string, std::string> args_map;
    std::string report_time;

    friend bool operator==(const ErrorItem &lhs, const ErrorItem &rhs) noexcept
    {
        return (lhs.error_id == rhs.error_id) && (lhs.error_message == rhs.error_message) &&
               (lhs.possible_cause == rhs.possible_cause) && (lhs.solution == rhs.solution);
    }
};
} // namespace error_message

class ErrorManager {
public:
    using ErrorItem = error_message::ErrorItem;
    /// @brief Obtain  ErrorManager instance
    /// @return ErrorManager instance
    static ErrorManager &GetInstance();

    /// @brief init
    /// @return int 0(success) -1(fail)
    int32_t Init();

    int32_t Init(error_message::ErrorMsgMode error_mode);

    /// @brief init
    /// @param [in] path: current so path
    /// @return int 0(success) -1(fail)
    int32_t Init(const std::string path);

    int32_t ReportInterErrMessage(const std::string error_code, const std::string &error_msg);

    /// @brief Report error message
    /// @param [in] error_code: error code
    /// @param [in] args_map: parameter map
    /// @return int 0(success) -1(fail)
    int32_t ReportErrMessage(const std::string error_code, const std::map<std::string, std::string> &args_map);

    /// @brief output error message
    /// @param [in] handle: print handle
    /// @return int 0(success) -1(fail)
    int32_t OutputErrMessage(int32_t handle);

    /// @brief output  message
    /// @param [in] handle: print handle
    /// @return int 0(success) -1(fail)
    int32_t OutputMessage(int32_t handle);
    // 调用成功后会进行已有的错误信息的清理
    std::string GetErrorMessage();

    // 获取当前线程下原始上报的内部，外部错误信息，顺序为上报的原始顺序
    // 便于调用方进行自定制的加工，调用成功后会进行已有的错误信息的清理
    std::vector<ErrorItem> GetRawErrorMessages();

    std::string GetWarningMessage();

    /// @brief Report error message
    /// @param [in] key: vector parameter key
    /// @param [in] value: vector parameter value
    void ATCReportErrMessage(const std::string error_code, const std::vector<std::string> &key = {},
                             const std::vector<std::string> &value = {});

    /// @brief report graph compile failed message such as error code and op_name in mstune case
    /// @param [in] graph_name: root graph name
    /// @param [in] msg: failed message map, key is error code, value is op_name
    /// @return int 0(success) -1(fail)
    int32_t ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
                                         const std::map<std::string, std::string> &msg);

    /// @brief get graph compile failed message in mstune case
    /// @param [in] graph_name: graph name
    /// @param [out] msg_map: failed message map, key is error code, value is op_name list
    /// @return int 0(success) -1(fail)
    int32_t GetMstuneCompileFailedMsg(const std::string &graph_name,
                                      std::map<std::string, std::vector<std::string>> &msg_map);

    // @brief generate work_stream_id by current pid and tid, clear error_message stored by same work_stream_id
    // used in external api entrance, all sync api can use
    void GenWorkStreamIdDefault();

    // @brief generate work_stream_id by args sessionid and graphid, clear error_message stored by same work_stream_id
    // used in external api entrance
    void GenWorkStreamIdBySessionGraph(const uint64_t session_id, const uint64_t graph_id);

    const std::string &GetLogHeader();

    error_message::Context &GetErrorManagerContext();

    void SetErrorContext(error_message::Context error_context);

    void SetStage(const std::string &first_stage, const std::string &second_stage);

private:
    struct ErrorInfoConfig {
        std::string error_id;
        std::string error_message;
        std::string possible_cause;
        std::string solution;
        std::vector<std::string> arg_list;
    };
    ErrorManager()  = default;
    ~ErrorManager() = default;

    ErrorManager(const ErrorManager &)              = delete;
    ErrorManager(ErrorManager &&)                   = delete;
    ErrorManager &operator=(const ErrorManager &) & = delete;
    ErrorManager &operator=(ErrorManager &&) &      = delete;

    int32_t ParseJsonFile(const std::string path);

    static int32_t ReadJsonFile(const std::string &file_path, void *const handle);

    void ClassifyCompileFailedMsg(const std::map<std::string, std::string> &msg,
                                  std::map<std::string, std::vector<std::string>> &classified_msg);

    bool IsInnerErrorCode(const std::string &error_code) const;

    bool IsParamCheckErrorId(const std::string &error_code) const;

    inline bool IsValidErrorCode(const std::string &error_codes) const
    {
        constexpr uint32_t kErrorCodeValidLength = 6U;
        return error_codes.size() == kErrorCodeValidLength;
    }

    std::vector<ErrorItem> &GetErrorMsgContainerByWorkId(uint64_t work_id);
    std::vector<ErrorItem> &GetWarningMsgContainerByWorkId(uint64_t work_id);

    std::vector<ErrorItem> &GetErrorMsgContainer(uint64_t work_stream_id);
    std::vector<ErrorItem> &GetWarningMsgContainer(uint64_t work_stream_id);

    void AssembleInnerErrorMessage(const std::vector<ErrorItem> &error_messages, const std::string &first_code,
                                   std::stringstream &err_stream) const;

    void ClearErrorMsgContainerByWorkId(const uint64_t work_stream_id);
    void ClearWarningMsgContainerByWorkId(const uint64_t work_stream_id);

    void ClearErrorMsgContainer(const uint64_t work_stream_id);
    void ClearWarningMsgContainer(const uint64_t work_stream_id);

    bool is_init_ = false;
    std::mutex mutex_;
    std::map<std::string, ErrorInfoConfig> error_map_;
    std::map<std::string, std::map<std::string, std::vector<std::string>>> compile_failed_msg_map_;

    std::map<uint64_t, std::vector<ErrorItem>> error_message_per_work_id_;
    std::map<uint64_t, std::vector<ErrorItem>> warning_messages_per_work_id_;

    thread_local static error_message::Context error_context_;

    error_message::ErrorMsgMode error_mode_ = error_message::ErrorMsgMode::INTERNAL_MODE;
    std::vector<ErrorItem> error_message_process_;    // 进程粒度，所有的errmsg存到同一个vector
    std::vector<ErrorItem> warning_messages_process_; // 进程粒度，所有的warning msg存到同一个vector
};
#endif // ERROR_MANAGER_H_
