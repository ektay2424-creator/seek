// src/main.cpp
#include "seek/chunker.hpp"
#include "seek/crawler.hpp"
#include "seek/embedder_factory.hpp"
#include "seek/fs_watcher.hpp"
#include "seek/index_store.hpp"
#include "seek/indexer.hpp"
#include "seek/tokenizer.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace seek;
namespace fs = std::filesystem;

static int cmd_index(const std::vector<std::string>& roots, const std::string& out,
                     const std::string& kind) {
    auto embedder = make_embedder(kind, 384);
    auto idx = build_index(roots, *embedder);
    std::cerr << "indexed " << idx.docs.size() << " docs, "
              << idx.chunks.size() << " chunks, vocab "
              << idx.inv.vocab_size() << " [embedder=" << kind << "]\n";
    if (!save_index(idx, out)) { std::cerr << "save failed\n"; return 1; }
    std::cerr << "saved -> " << out << "\n";
    return 0;
}

static int cmd_query(const std::string& q, const std::string& index_path,
                     size_t top_k, const std::string& kind) {
    SeekIndex idx;
    if (!load_index(idx, index_path)) { std::cerr << "load failed\n"; return 1; }
    if (q.empty())                     { std::cerr << "empty query\n"; return 1; }

    auto terms   = tokenize(q);
    auto lexical = idx.inv.search(terms, top_k * 4);

    auto embedder = make_embedder(kind, 384);
    auto qvec     = embedder->embed(q);
    auto semantic = idx.vec.search(qvec, top_k * 4);

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

static int cmd_autocomplete(const std::string& prefix, const std::string& index_path) {
    SeekIndex idx;
    if (!load_index(idx, index_path)) { std::cerr << "load failed\n"; return 1; }
    for (auto& w : idx.trie.complete(prefix, 10)) std::cout << w << "\n";
    return 0;
}

struct WatchEngine {
    SeekIndex idx;
    Embedder* embedder = nullptr;
    struct Cached { uint64_t hash; DocId doc_id; std::vector<Chunk> chunks; };
    std::map<std::string, Cached> files;
    DocId next_doc = 0;

    void ingest(const std::string& path) {
        std::string text = read_file(path);
        if (text.empty()) { files.erase(path); return; }
        auto it = files.find(path);
        DocId did = (it == files.end()) ? next_doc++ : it->second.doc_id;
        auto chunks = chunk_document(did, text);
        uint64_t h  = static_cast<uint64_t>(std::hash<std::string>{}(text));
        files[path] = Cached{h, did, std::move(chunks)};
    }
    void forget(const std::string& path) { files.erase(path); }

    void rebuild() {
        idx = SeekIndex{};
        ChunkId next_chunk = 0;
        for (auto& [path, cf] : files) {
            idx.docs.push_back({cf.doc_id, path, cf.hash});
            for (auto& c : cf.chunks) {
                Chunk copy = c;
                copy.id     = next_chunk++;
                copy.doc_id = cf.doc_id;
                idx.inv.add_chunk(copy.id, copy.text);
                for (auto& t : tokenize(copy.text)) idx.trie.insert(t);
                idx.vec.add(copy.id, embedder->embed(copy.text));
                idx.chunks.push_back(std::move(copy));
            }
        }
    }
};

static int cmd_watch(const std::vector<std::string>& roots, const std::string& out,
                     int interval, const std::string& kind) {
    if (roots.empty()) { std::cerr << "watch needs a directory\n"; return 1; }

    auto embedder = make_embedder(kind, 384);
    WatchEngine eng;
    eng.embedder = embedder.get();
    PollingWatcher watcher(roots);

    for (auto& p : watcher.scan()) eng.ingest(p);
    eng.rebuild();
    save_index(eng.idx, out);
    std::cerr << "watching " << roots.size() << " root(s), "
              << eng.files.size() << " files, "
              << eng.idx.chunks.size() << " chunks [embedder=" << kind
              << "]. Ctrl-C to stop.\n";

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(interval));
        auto changes = watcher.scan();
        if (changes.empty()) continue;
        std::cerr << "[" << changes.size() << " change(s)]\n";
        for (auto& p : changes) {
            if (fs::exists(p)) eng.ingest(p);
            else                eng.forget(p);
        }
        eng.rebuild();
        save_index(eng.idx, out);
        std::cerr << "  reindexed: " << eng.files.size() << " files, "
                  << eng.idx.chunks.size() << " chunks\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage:\n"
                  << "  seek index <dir...> [--out .seekidx] [--embedder hash|onnx]\n"
                  << "  seek query \"<q>\" [--index .seekidx] [--top 5] [--embedder hash|onnx]\n"
                  << "  seek autocomplete <prefix> [--index .seekidx]\n"
                  << "  seek watch <dir...> [--index .seekidx] [--interval 2] [--embedder hash|onnx]\n";
        return 1;
    }

    std::string cmd = argv[1];
    std::string idx_path = ".seekidx";
    std::string embedder_kind = "hash";
    size_t top_k = 5;
    int interval = 2;
    std::vector<std::string> pos;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--out"      && i + 1 < argc) idx_path       = argv[++i];
        else if (a == "--index"    && i + 1 < argc) idx_path       = argv[++i];
        else if (a == "--top"      && i + 1 < argc) top_k          = (size_t)std::stoul(argv[++i]);
        else if (a == "--interval" && i + 1 < argc) interval       = std::stoi(argv[++i]);
        else if (a == "--embedder" && i + 1 < argc) embedder_kind  = argv[++i];
        else                                        pos.push_back(a);
    }

    if (cmd == "index")        return cmd_index(pos, idx_path, embedder_kind);
    if (cmd == "query")        return cmd_query(pos.empty() ? "" : pos[0], idx_path, top_k, embedder_kind);
    if (cmd == "autocomplete") return cmd_autocomplete(pos.empty() ? "" : pos[0], idx_path);
    if (cmd == "watch")        return cmd_watch(pos, idx_path, interval, embedder_kind);

    std::cerr << "unknown command: " << cmd << "\n";
    return 1;
}