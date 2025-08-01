#pragma once
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include <string>
#include <iostream>

namespace deep_ep {
int get_value_from_env(const std::string &name, int defaultValue)
{
    int retValue = defaultValue;
    if (const char* rank_str = std::getenv(name.c_str())) {
        char* end;
        errno = 0;
        long val = std::strtol(rank_str, &end, 10);
        if (errno == ERANGE || *end != '\0' || !std::isdigit(*rank_str)) {
            std::cerr << "Warning: The environment variable " << name << " is not valid.\n";
            return retValue;
        }
        retValue = static_cast<int>(val);
        return retValue;
    } else {
        std::cerr << "Warning: The environment variable " << name << " is not set.\n";
        return retValue;
    }
}

} // namespace deep_ep