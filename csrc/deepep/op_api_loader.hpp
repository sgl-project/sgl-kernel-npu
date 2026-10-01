#ifndef DEEP_EP_OP_API_LOADER_HPP
#define DEEP_EP_OP_API_LOADER_HPP

#include <dlfcn.h>
#include <unistd.h>
#include <string>

namespace deep_ep::op_api {
struct Library {
    void *handle;
    std::string path;
    std::string error;
};

inline Library OpenLibrary(const std::string &path)
{
    auto handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    const char *error = handle == nullptr ? dlerror() : nullptr;
    return {handle, path, error == nullptr ? "" : error};
}

// Internal linkage keeps the address anchored in deep_ep_cpp even if another
// extension includes a similar loader. Python imports extension modules by path.
static inline std::string BundledLibraryPath()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&BundledLibraryPath), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    std::string extensionPath(info.dli_fname);
    auto slash = extensionPath.find_last_of('/');
    if (slash == std::string::npos) {
        return {};
    }
    return extensionPath.substr(0, slash) + "/vendors/hwcomputing/op_api/lib/libcust_opapi.so";
}

inline const Library &CustomLibrary()
{
    static const Library library = [] {
        auto path = BundledLibraryPath();
        if (!path.empty() && access(path.c_str(), F_OK) == 0) {
            // Do not fall back to an older external custom library if this
            // wheel's library exists but cannot load. Retain its dlerror.
            return OpenLibrary(path);
        }
        // Support development builds that install the custom OPP separately.
        return OpenLibrary("libcust_opapi.so");
    }();
    return library;
}

inline const Library &SystemLibrary()
{
    static const Library library = OpenLibrary("libopapi.so");
    return library;
}

inline void *FindFunction(const char *name)
{
    if (CustomLibrary().handle != nullptr) {
        if (auto address = dlsym(CustomLibrary().handle, name)) {
            return address;
        }
    }
    return SystemLibrary().handle == nullptr ? nullptr : dlsym(SystemLibrary().handle, name);
}

inline std::string Describe(const Library &library)
{
    return library.path + (library.handle != nullptr ? " (loaded)" : " (load failed: " + library.error + ")");
}
}  // namespace deep_ep::op_api

#endif
