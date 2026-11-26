#pragma once
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <vector>

namespace seek {

class VectorIndex {
public:
    void add(ChunkId id, std::vector<float> emb) {
        normalize(emb);
        ids_.push_back(id);
        dim_ = emb.size();
        data_.insert(data_.end(), emb.begin(), emb.end());
    }

    std::vector<ScoredChunk> search(std::vector<float> q, size_t top_k = 10) const {
        if (dim_ == 0) return {};
        normalize(q);

        auto cmp = [](const ScoredChunk& a, const ScoredChunk& b) { return a.score > b.score; };
        std::priority_queue<ScoredChunk, std::vector<ScoredChunk>, decltype(cmp)> heap(cmp); // min-heap

        for (size_t i = 0; i < ids_.size(); ++i) {
            const float* row = data_.data() + i * dim_;
            float dot = 0;
            for (size_t d = 0; d < dim_; ++d) dot += row[d] * q[d]; // cosine == dot (normalized)
            if (heap.size() < top_k) heap.push({ids_[i], dot});
            else if (dot > heap.top().score) { heap.pop(); heap.push({ids_[i], dot}); }
        }
        std::vector<ScoredChunk> out;
        while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
        std::reverse(out.begin(), out.end());   // best first
        return out;
    }

    size_t size() const { return ids_.size(); }
    size_t dim()  const { return dim_; }
    const std::vector<ChunkId>& ids()  const { return ids_; }
    const std::vector<float>&   data() const { return data_; }

private:
    static void normalize(std::vector<float>& v) {
        float n = 0;
        for (float x : v) n += x * x;
        n = std::sqrt(n);
        if (n > 1e-9f) for (float& x : v) x /= n;
    }
    std::vector<ChunkId> ids_;
    std::vector<float>   data_;
    size_t dim_ = 0;
};

// Reciprocal Rank Fusion
inline std::vector<ScoredChunk> rrf_fuse(const std::vector<std::vector<ScoredChunk>>& lists,
                                         size_t top_k, float k = 60.0f) {
    std::unordered_map<ChunkId, float> fused;
    for (auto& l : lists)
        for (size_t r = 0; r < l.size(); ++r)
            fused[l[r].id] += 1.0f / (k + float(r + 1));
    std::vector<ScoredChunk> out;
    out.reserve(fused.size());
    for (auto& [id, s] : fused) out.push_back({id, s});
    size_t k2 = std::min(top_k, out.size());
    std::partial_sort(out.begin(), out.begin() + k2, out.end(),
                      [](auto& a, auto& b) { return a.score > b.score; });
    out.resize(k2);
    return out;
}
} // namespace seek