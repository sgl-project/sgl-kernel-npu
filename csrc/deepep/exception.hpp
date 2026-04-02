#pragma once

#include <string>
#include <exception>

namespace deep_ep {

class EPException : public std::exception
{
private:
    std::string message = {};

public:
    explicit EPException(const char *name, const char *file, const int line, const std::string &error)
    {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) +
                  " error message or error code is '" + error + "'";
    }

    const char *what() const noexcept override
    {
        return message.c_str();
    }
};

}  // namespace deep_ep

template <typename... Args>
void log_all(std::ostream &os, Args &&...args)
{
    (os << ... << std::forward<Args>(args));
}

#define EP_HOST_ASSERT(cond)                                           \
    ;                                                                  \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)

#define EP_HOST_ASSERT_S(cond, ...)                                        \
    do {                                                                   \
        if (not(cond)) {                                                   \
            std::ostringstream oss;                                        \
            oss << "(" #cond ") ";                                         \
            log_all(oss, __VA_ARGS__);                                     \
            oss << std::endl;                                              \
            throw EPException("Assertion", __FILE__, __LINE__, oss.str()); \
        }                                                                  \
    } while (0)

#define ACL_CHECK(ret)                                                                            \
    ;                                                                                             \
    do {                                                                                          \
        if (ret != ACL_SUCCESS) {                                                                 \
            throw deep_ep::EPException("ACL Assertion", __FILE__, __LINE__, std::to_string(ret)); \
        }                                                                                         \
    } while (0)

#define HCCL_CHECK(ret)                                                                            \
    ;                                                                                              \
    do {                                                                                           \
        if (ret != HCCL_SUCCESS) {                                                                 \
            throw deep_ep::EPException("HCCL Assertion", __FILE__, __LINE__, std::to_string(ret)); \
        }                                                                                          \
    } while (0)

#ifdef DEBUG_MODE
#define LOG_DEBUG(msg)                                \
    do {                                              \
        std::cout << "DEBUG: " << (msg) << std::endl; \
    } while (0)
#else
#define LOG_DEBUG(msg) \
    do {               \
    } while (0)
#endif
