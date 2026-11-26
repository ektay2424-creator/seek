#pragma once
#include "embedder.hpp"
#include "onnx_loader.hpp"
#include "wordpiece.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace seek {

class OnnxEmbedder : public Embedder {
public:
    OnnxEmbedder(const std::string& model_path, const std::string& vocab_path)
        : tok_(vocab_path) {
        const OrtApi* api = OnnxRuntime::api();
        ORT(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "seek", &env_));

        // Read model into memory -> avoids Windows wchar_t path issue entirely.
        std::ifstream mf(model_path, std::ios::binary | std::ios::ate);
        if (!mf) throw std::runtime_error("Cannot open model: " + model_path);
        auto sz = mf.tellg();
        if (sz <= 0) throw std::runtime_error("Empty model file: " + model_path);
        mf.seekg(0);
        std::vector<char> model_buf(static_cast<size_t>(sz));
        mf.read(model_buf.data(), sz);

        OrtSessionOptions* opts = nullptr;
        ORT(api->CreateSessionOptions(&opts));
        options_ = opts;
        ORT(api->SetIntraOpNumThreads(opts, 2));
        ORT(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
        ORT(api->CreateSessionFromArray(env_, model_buf.data(), model_buf.size(),
                                        opts, &session_));

        ORT(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_));

        OrtAllocator* alloc = nullptr;
        ORT(api->GetAllocatorWithDefaultOptions(&alloc));
        alloc_ = alloc;

        size_t n_inputs = 0;
        ORT(api->SessionGetInputCount(session_, &n_inputs));
        for (size_t i = 0; i < n_inputs; ++i) {
            char* name = nullptr;
            ORT(api->SessionGetInputName(session_, i, alloc, &name));
            std::string s(name);
            alloc->Free(alloc, name);
            if      (s == "input_ids")      has_ids_   = true;
            else if (s == "attention_mask") has_mask_  = true;
            else if (s == "token_type_ids") has_ttids_ = true;
        }
        if (!has_ids_) throw std::runtime_error("ONNX model missing input_ids input");

        size_t n_out = 0;
        ORT(api->SessionGetOutputCount(session_, &n_out));
        if (n_out < 1) throw std::runtime_error("ONNX model has no outputs");
        char* out_name = nullptr;
        ORT(api->SessionGetOutputName(session_, 0, alloc, &out_name));
        output_name_ = out_name;
        alloc->Free(alloc, out_name);

        dim_ = 384;
    }

    ~OnnxEmbedder() override {
        if (mem_info_) OnnxRuntime::api()->ReleaseMemoryInfo(mem_info_);
        if (session_)  OnnxRuntime::api()->ReleaseSession(session_);
        if (options_)  OnnxRuntime::api()->ReleaseSessionOptions(options_);
        if (env_)      OnnxRuntime::api()->ReleaseEnv(env_);
    }

    std::vector<float> embed(const std::string& text) override {
        const OrtApi* api = OnnxRuntime::api();
        auto [ids, mask] = tok_.encode(text);
        int64_t seq = (int64_t)ids.size();

        std::vector<int64_t> shape = {1, seq};

        OrtValue* ids_val = nullptr;
        ORT(api->CreateTensorWithDataAsOrtValue(
            mem_info_, ids.data(), ids.size() * sizeof(int64_t),
            shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ids_val));

        OrtValue* mask_val = nullptr;
        ORT(api->CreateTensorWithDataAsOrtValue(
            mem_info_, mask.data(), mask.size() * sizeof(int64_t),
            shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &mask_val));

        std::vector<int64_t> ttids(ids.size(), 0);
        OrtValue* tt_val = nullptr;
        std::vector<const char*> names;
        std::vector<OrtValue*>    vals;

        names.push_back("input_ids");      vals.push_back(ids_val);
        if (has_mask_)  { names.push_back("attention_mask"); vals.push_back(mask_val); }
        if (has_ttids_) {
            ORT(api->CreateTensorWithDataAsOrtValue(
                mem_info_, ttids.data(), ttids.size() * sizeof(int64_t),
                shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &tt_val));
            names.push_back("token_type_ids"); vals.push_back(tt_val);
        }

        const char* out_names[] = {output_name_.c_str()};
        OrtValue* out = nullptr;
        OrtStatus* st = api->Run(session_, nullptr,
                                 names.data(), vals.data(), names.size(),
                                 out_names, 1, &out);

        api->ReleaseValue(ids_val);
        api->ReleaseValue(mask_val);
        if (tt_val) api->ReleaseValue(tt_val);

        if (st) {
            const char* m = api->GetErrorMessage(st);
            std::string msg = m ? m : "unknown";
            api->ReleaseStatus(st);
            throw std::runtime_error("ONNX Run failed: " + msg);
        }

        float* raw = nullptr;
        ORT(api->GetTensorMutableData(out, reinterpret_cast<void**>(&raw)));

        std::vector<float> v(dim_);
        for (size_t i = 0; i < dim_; ++i) v[i] = raw[i];
        api->ReleaseValue(out);

        float n = 0;
        for (float x : v) n += x * x;
        n = std::sqrt(n);
        if (n > 1e-9f) for (float& x : v) x /= n;
        return v;
    }

    size_t dim() const override { return dim_; }

private:
    WordPieceTokenizer tok_;
    OrtEnv*            env_      = nullptr;
    OrtSession*        session_  = nullptr;
    OrtSessionOptions* options_  = nullptr;
    OrtMemoryInfo*     mem_info_ = nullptr;
    OrtAllocator*      alloc_    = nullptr;
    std::string        output_name_;
    bool  has_ids_ = false, has_mask_ = false, has_ttids_ = false;
    size_t dim_ = 384;
};
} // namespace seek