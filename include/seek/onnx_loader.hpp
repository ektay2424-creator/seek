#pragma once
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "onnxruntime_c_api.h"

#include <stdexcept>
#include <string>

namespace seek {

class OnnxRuntime {
public:
    static const OrtApi* api() {
        static OnnxRuntime instance;
        return instance.api_;
    }

private:
    OnnxRuntime() {
#ifdef _WIN32
        HMODULE h = LoadLibraryA("onnxruntime.dll");
        if (!h) {
            throw std::runtime_error(
                "Failed to load onnxruntime.dll. "
                "Copy third_party/onnxruntime/.../lib/onnxruntime.dll next to "
                "seek.exe (build/), or add its folder to PATH.");
        }
        using GetApiBaseFn = const OrtApiBase* (*)();
        auto getApiBase = reinterpret_cast<GetApiBaseFn>(
            reinterpret_cast<void*>(GetProcAddress(h, "OrtGetApiBase")));
        if (!getApiBase) {
            throw std::runtime_error("onnxruntime.dll missing OrtGetApiBase symbol.");
        }
#else
        using GetApiBaseFn = const OrtApiBase* (*)();
        auto getApiBase = reinterpret_cast<GetApiBaseFn>(
            reinterpret_cast<void*>(dlsym(RTLD_DEFAULT, "OrtGetApiBase")));
        if (!getApiBase) throw std::runtime_error("libonnxruntime not preloaded.");
#endif
        const OrtApiBase* base = getApiBase();
        api_ = base->GetApi(ORT_API_VERSION);
        if (!api_) throw std::runtime_error("ONNX Runtime API version mismatch.");
    }

    const OrtApi* api_ = nullptr;
};

#define ORT(x) do { \
    OrtStatus* _s = (x); \
    if (_s) { \
        const char* m = OnnxRuntime::api()->GetErrorMessage(_s); \
        std::string msg = m ? m : "unknown onnx error"; \
        OnnxRuntime::api()->ReleaseStatus(_s); \
        throw std::runtime_error("ONNX error: " + msg); \
    } \
} while (0)

} // namespace seek