// src/main.cpp
#include "seek/chunker.hpp"
#include "seek/crawler.hpp"
#include "seek/embedder.hpp"
#include "seek/index_store.hpp"
#include "seek/thread_pool.hpp"
#include "seek/tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace seek;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// seek index <dir...> [--out .seekidx]
// ---------------------------------------------------------------------------
static int cmd_index(const std::vector<std::string>& roots, const std::string& out) {
    auto files = crawl(roots);
    std::cerr << "crawled " << files.size() << " files\n";

    SeekIndex idx;
    auto embedder = make_embedder("hash", 384);   // <-- swap to "onnx" later
    ThreadPool pool;

    std::mutex mu;
    DocId   next_doc   = 0;
    ChunkId next_chunk = 0;

    for (auto& path : files) {
        pool.enqueue([&, path] {
            std::string text = read_file(path.string());
            if (text.empty()) return;

            // Heavy work (parsing, chunking, tokenizing, embedding) is lock-free.
            auto chunks = chunk_document(0, text);   // doc_id patched below
            std::vector<std::vector<float>> embs;
            embs.reserve(chunks.size());
            for (auto& c : chunks) embs.push_back(embedder->embed(c.text));

            // ---- critical section: mutate the shared index ----
            std::lock_guard<std::mutex> lk(mu);

            DocId did = next_doc++;
            uint64_t hash = static_cast<uint64_t>(std::hash<std::string>{}(text));
            idx.docs.push_back({did, path.string(), hash});

            for (size_t i = 0; i < chunks.size(); ++i) {
                auto& c   = chunks[i];
                c.id      = next_chunk++;
                c.doc_id  = did;

                idx.inv.add_chunk(c.id, c.text);
                for (auto& t : tokenize(c.text)) idx.trie.insert(t);
                idx.vec.add(c.id, std::move(embs[i]));

                idx.chunks.push_back(std::move(c));
            }
        });
    }
    pool.wait_idle();
    pool.stop();

    std::cerr << "indexed " << idx.chunks.size() << " chunks, vocab "
              << idx.inv.vocab_size() << "\n";

    if (!save_index(idx, out)) { std::cerr << "save failed\n"; return 1; }
    std::cerr << "saved -> " << out << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// seek query "<q>" [--index .seekidx] [--top 5]
// Hybrid: BM25 lexical + dense vector cosine -> Reciprocal Rank Fusion.
// ---------------------------------------------------------------------------
static int cmd_query(const std::string& q, const std::string& index_path, size_t top_k) {
    SeekIndex idx;
    if (!load_index(idx, index_path)) { std::cerr << "load failed\n"; return 1; }
    if (q.empty())                     { std::cerr << "empty query\n"; return 1; }

    // 1. lexical (BM25) — pull a wider net than top_k so RRF has material to work with
    auto terms   = tokenize(q);
    auto lexical = idx.inv.search(terms, top_k * 4);

    // 2. semantic (vector cosine)
    auto embedder = make_embedder("hash", 384);
    auto qvec     = embedder->embed(q);
    auto semantic = idx.vec.search(qvec, top_k * 4);

    // 3. fuse
    auto hits = rrf_fuse({lexical, semantic}, top_k);

    for (auto& h : hits) {
        auto& c = idx.chunks[h.id];
        auto& d = idx.docs[c.doc_id];
        std::cout << "[" << h.score << "] " << d.path << ":" << c.start_line << "\n";

        std::string snip = c.text.substr(0, 160);
        for (auto& ch : snip) if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        std::cout << "    " << snip << "...\n\n";
    }
    if (hits.empty()) std::cout << "(no matches)\n";
    return 0;
}

// ---------------------------------------------------------------------------
// seek autocomplete <prefix> [--index .seekidx]
// ---------------------------------------------------------------------------
static int cmd_autocomplete(const std::string& prefix, const std::string& index_path) {
    SeekIndex idx;
    if (!load_index(idx, index_path)) { std::cerr << "load failed\n"; return 1; }
    for (auto& w : idx.trie.complete(prefix, 10)) std::cout << w << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage:\n"
                  << "  seek index <dir...> [--out .seekidx]\n"
                  << "  seek query \"<q>\" [--index .seekidx] [--top 5]\n"
                  << "  seek autocomplete <prefix> [--index .seekidx]\n";
        return 1;
    }

    std::string cmd = argv[1];
    std::string idx_path = ".seekidx";
    size_t top_k = 5;
    std::vector<std::string> pos;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--out"   && i + 1 < argc) idx_path = argv[++i];
        else if (a == "--index" && i + 1 < argc) idx_path = argv[++i];
        else if (a == "--top"   && i + 1 < argc) top_k    = static_cast<size_t>(std::stoul(argv[++i]));
        else                                     pos.push_back(a);
    }

    if (cmd == "index")        return cmd_index(pos, idx_path);
    if (cmd == "query")        return cmd_query(pos.empty() ? "" : pos[0], idx_path, top_k);
    if (cmd == "autocomplete") return cmd_autocomplete(pos.empty() ? "" : pos[0], idx_path);

    std::cerr << "unknown command: " << cmd << "\n";
    return 1;
}