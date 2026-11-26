#pragma once
#include "tokenizer.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace seek {

class Embedder {
public:
    virtual ~Embedder() = default;
    virtual std::vector<float> embed(const std::string& text) = 0;
    virtual size_t dim() const = 0;
};

class HashEmbedder : public Embedder {
public:
    explicit HashEmbedder(size_t dim = 384) : dim_(dim) {}

    std::vector<float> embed(const std::string& text) override {
        std::vector<float> v(dim_, 0.0f);
        for (auto& tok : tokenize(text)) {
            uint64_t h    = fnv1a(tok);
            size_t   slot = static_cast<size_t>(h % dim_);
            float    sign = (h >> 63) & 1 ? -1.0f : 1.0f;
            v[slot] += sign;
        }
        float n = 0.0f;
        for (float x : v) n += x * x;
        n = std::sqrt(n);
        if (n > 1e-9f) for (float& x : v) x /= n;
        return v;
    }

    size_t dim() const override { return dim_; }

private:
    static uint64_t fnv1a(const std::string& s) {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    }
    size_t dim_;
};

} // namespace seek