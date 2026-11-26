#pragma once
#include "chunker.hpp"
#include "crawler.hpp"
#include "embedder.hpp"
#include "index_store.hpp"
#include "thread_pool.hpp"
#include "tokenizer.hpp"

#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace seek {

inline std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

inline SeekIndex build_index(const std::vector<std::string>& roots, Embedder& embedder,
                             size_t n_threads = 0) {
    auto files = crawl(roots);
    SeekIndex idx;
    ThreadPool pool(n_threads);
    std::mutex mu;
    DocId   next_doc   = 0;
    ChunkId next_chunk = 0;

    for (auto& path : files) {
        pool.enqueue([&, path] {
            std::string text = read_file(path.string());
            if (text.empty()) return;
            auto chunks = chunk_document(0, text);
            std::vector<std::vector<float>> embs;
            embs.reserve(chunks.size());
            for (auto& c : chunks) embs.push_back(embedder.embed(c.text));

            std::lock_guard<std::mutex> lk(mu);
            DocId did = next_doc++;
            uint64_t h = static_cast<uint64_t>(std::hash<std::string>{}(text));
            idx.docs.push_back({did, path.string(), h});
            for (size_t i = 0; i < chunks.size(); ++i) {
                auto& c = chunks[i];
                c.id     = next_chunk++;
                c.doc_id = did;
                idx.inv.add_chunk(c.id, c.text);
                for (auto& t : tokenize(c.text)) idx.trie.insert(t);
                idx.vec.add(c.id, std::move(embs[i]));
                idx.chunks.push_back(std::move(c));
            }
        });
    }
    pool.wait_idle();
    pool.stop();
    return idx;
}
} // namespace seek