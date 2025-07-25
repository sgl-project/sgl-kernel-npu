#pragma once

#include <string>
#include <exception>

namespace deep_ep {

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char *name, const char *file, const int line, const std::string &error)
    {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) +
                  " error meessage or error code is '" + error + "'";
    }

    const char *what() const noexcept override
    {
        return message.c_str();
    }
};

}

#define EP_HOST_ASSERT(cond); \
do { \
    if (not (cond)) { \
        throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)


#define ACL_CHECK(ret); \
do { \
    if(ret != ACL_SUCCESS)\
    { \
        throw deep_ep::EPException("ACL Assertion", __FILE__, __LINE__, std::to_string(ret));\
    } \
} while (0)

#define HCCL_CHECK(ret); \
do { \
    if(ret != HCCL_SUCCESS) \
    {   \
        throw deep_ep::EPException("HCCL Assertion", __FILE__, __LINE__, std::to_string(ret)); \
    } \
} while (0)
