#pragma once
#include "types.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace seek {

struct Posting { ChunkId chunk_id; uint32_t tf; };

class InvertedIndex {
public:
    void add_chunk(ChunkId id, const std::string& text) {
        auto toks = tokenize(text);
        std::unordered_map<std::string, uint32_t> tf;
        for (auto& t : toks) tf[t]++;
        for (auto& [term, freq] : tf) index_[term].push_back({id, freq});
        chunk_len_.push_back(static_cast<uint32_t>(toks.size()));
        total_len_ += toks.size();
        num_chunks_ = chunk_len_.size();
        avg_len_ = num_chunks_ ? double(total_len_) / double(num_chunks_) : 0.0;
    }

    const std::vector<Posting>* postings(std::string_view term) const {
        auto it = index_.find(std::string(term));
        return it == index_.end() ? nullptr : &it->second;
    }

    void score_term(std::string_view term, std::vector<float>& scores,
                    float k1 = 1.5f, float b = 0.75f) const {
        const auto* pl = postings(term);
        if (!pl || pl->empty()) return;
        double N = double(num_chunks_);
        double idf = std::log(1.0 + (N - double(pl->size()) + 0.5) / (double(pl->size()) + 0.5));
        double avg = avg_len_ > 0 ? avg_len_ : 1.0;
        for (auto& p : *pl) {
            double ln = 1.0 - b + b * (double(chunk_len_[p.chunk_id]) / avg);
            double denom = double(p.tf) + k1 * ln;
            double s = idf * (double(p.tf) * (k1 + 1.0)) / (denom > 0 ? denom : 1.0);
            scores[p.chunk_id] += float(s);
        }
    }

    std::vector<ScoredChunk> search(const std::vector<std::string>& terms, size_t top_k = 10) const {
        std::vector<float> scores(num_chunks_, 0.f);
        for (auto& t : terms) score_term(t, scores);
        std::vector<ScoredChunk> out;
        for (ChunkId i = 0; i < scores.size(); ++i)
            if (scores[i] > 0) out.push_back({i, scores[i]});
        size_t k = std::min(top_k, out.size());
        std::partial_sort(out.begin(), out.begin() + k, out.end(),
                          [](auto& a, auto& b) { return a.score > b.score; });
        out.resize(k);
        return out;
    }

    size_t num_chunks() const { return num_chunks_; }
    size_t vocab_size() const { return index_.size(); }
    const auto& raw()          const { return index_; }
    const auto& chunk_lengths() const { return chunk_len_; }

    // Used by the loader
    void reserve_chunks(size_t n)                { chunk_len_.assign(n, 0); }
    void set_chunk_len(ChunkId id, uint32_t len) { chunk_len_[id] = len; total_len_ += len; }
    void add_posting(const std::string& t, Posting p) { index_[t].push_back(p); }
    void finalize() {
        num_chunks_ = chunk_len_.size();
        avg_len_ = num_chunks_ ? double(total_len_) / double(num_chunks_) : 0.0;
    }

private:
    std::unordered_map<std::string, std::vector<Posting>> index_;
    std::vector<uint32_t> chunk_len_;
    size_t total_len_ = 0, num_chunks_ = 0;
    double avg_len_ = 0.0;
};
} // namespace seek