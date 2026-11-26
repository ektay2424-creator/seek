#pragma once
#include "inverted_index.hpp"
#include "trie.hpp"
#include "types.hpp"
#include "vector_index.hpp"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace seek {

struct SeekIndex {
    std::vector<Document> docs;
    std::vector<Chunk>    chunks;
    InvertedIndex         inv;
    Trie                  trie;
    VectorIndex           vec;
};

// ---- primitives ----
inline void W32(std::ofstream& o, uint32_t v) { o.write((const char*)&v, 4); }
inline void W64(std::ofstream& o, uint64_t v) { o.write((const char*)&v, 8); }
inline void WStr(std::ofstream& o, const std::string& s) { W32(o, (uint32_t)s.size()); o.write(s.data(), s.size()); }
inline uint32_t R32(std::ifstream& i) { uint32_t v; i.read((char*)&v, 4); return v; }
inline uint64_t R64(std::ifstream& i) { uint64_t v; i.read((char*)&v, 8); return v; }
inline std::string RStr(std::ifstream& i) { uint32_t n = R32(i); std::string s(n, '\0'); i.read(s.data(), n); return s; }

inline constexpr char kMagic[8] = {'S','E','E','K','I','D','X','1'};

inline bool save_index(const SeekIndex& idx, const std::string& path) {
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    o.write(kMagic, 8);

    W32(o, (uint32_t)idx.docs.size());
    for (auto& d : idx.docs) { W32(o, d.id); WStr(o, d.path); W64(o, d.content_hash); }

    W32(o, (uint32_t)idx.chunks.size());
    for (auto& c : idx.chunks) { W32(o, c.id); W32(o, c.doc_id); W32(o, c.start_line); WStr(o, c.text); }

    auto& raw = idx.inv.raw();
    W32(o, (uint32_t)raw.size());
    for (auto& [term, plist] : raw) {
        WStr(o, term);
        W32(o, (uint32_t)plist.size());
        for (auto& p : plist) { W32(o, p.chunk_id); W32(o, p.tf); }
    }
    auto& cl = idx.inv.chunk_lengths();
    W32(o, (uint32_t)cl.size());
    for (auto v : cl) W32(o, v);

    W32(o, (uint32_t)idx.vec.size());
    W32(o, (uint32_t)idx.vec.dim());
    for (auto id : idx.vec.ids()) W32(o, id);
    auto& vd = idx.vec.data();
    o.write((const char*)vd.data(), vd.size() * sizeof(float));

    return (bool)o;
}

inline bool load_index(SeekIndex& idx, const std::string& path) {
    std::ifstream i(path, std::ios::binary);
    if (!i) return false;
    char magic[8]; i.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0) return false;

    idx.docs.resize(R32(i));
    for (auto& d : idx.docs) { d.id = R32(i); d.path = RStr(i); d.content_hash = R64(i); }

    idx.chunks.resize(R32(i));
    for (auto& c : idx.chunks) { c.id = R32(i); c.doc_id = R32(i); c.start_line = R32(i); c.text = RStr(i); }

    struct TP { std::string term; std::vector<Posting> postings; };
    std::vector<TP> terms(R32(i));
    for (auto& tp : terms) {
        tp.term = RStr(i);
        tp.postings.resize(R32(i));
        for (auto& p : tp.postings) { p.chunk_id = R32(i); p.tf = R32(i); }
    }
    uint32_t ncl = R32(i);
    idx.inv.reserve_chunks(ncl);
    for (uint32_t c = 0; c < ncl; ++c) idx.inv.set_chunk_len(c, R32(i));
    for (auto& tp : terms) { for (auto& p : tp.postings) idx.inv.add_posting(tp.term, p); idx.trie.insert(tp.term); }
    idx.inv.finalize();

    uint32_t nv = R32(i), dim = R32(i);
    std::vector<ChunkId> ids(nv);
    for (auto& v : ids) v = R32(i);
    std::vector<float> data(nv * dim);
    i.read((char*)data.data(), data.size() * sizeof(float));
    for (uint32_t k = 0; k < nv; ++k)
        idx.vec.add(ids[k], std::vector<float>(data.begin() + k * dim, data.begin() + (k + 1) * dim));

    return (bool)i;
}
} // namespace seek