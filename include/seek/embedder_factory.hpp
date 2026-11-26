#pragma once
#include "embedder.hpp"
#include "onnx_embedder.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace seek {

namespace detail {
inline std::string first_existing(const std::vector<std::string>& paths) {
    for (auto& p : paths) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) return p;
    }
    return {};
}
} // namespace detail

inline std::shared_ptr<Embedder> make_embedder(const std::string& kind, size_t dim = 384) {
    if (kind == "hash") return std::make_shared<HashEmbedder>(dim);

    if (kind == "onnx") {
        std::string model, vocab;
        if (const char* e = std::getenv("SEEK_ONNX_MODEL")) model = e;
        if (const char* e = std::getenv("SEEK_ONNX_VOCAB")) vocab = e;
        if (model.empty())
            model = detail::first_existing({"models/bge-small-en-v1.5.onnx",
                                            "../models/bge-small-en-v1.5.onnx"});
        if (vocab.empty())
            vocab = detail::first_existing({"models/vocab.txt", "../models/vocab.txt"});
        if (model.empty() || vocab.empty())
            throw std::runtime_error(
                "ONNX model/vocab not found. Place at models/bge-small-en-v1.5.onnx "
                "and models/vocab.txt, or set SEEK_ONNX_MODEL / SEEK_ONNX_VOCAB.");
        return std::make_shared<OnnxEmbedder>(model, vocab);
    }

    throw std::runtime_error("unknown embedder: " + kind);
}
} // namespace seek