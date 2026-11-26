#pragma once
#include "types.hpp"
#include "tokenizer.hpp"
#include <sstream>
#include <string>
#include <vector>

namespace seek {

struct ChunkConfig { size_t tokens_per_chunk = 256; size_t overlap_lines = 3; };

inline std::vector<Chunk> chunk_document(DocId doc_id, const std::string& text,
                                         const ChunkConfig& cfg = {}) {
    std::vector<std::string> lines;
    { std::istringstream ss(text); std::string l; while (std::getline(ss, l)) lines.push_back(l); }

    std::vector<Chunk> chunks;
    ChunkId next = 0;
    size_t i = 0;
    while (i < lines.size()) {
        size_t start = i, tok = 0;
        std::string buf;
        while (i < lines.size() && tok < cfg.tokens_per_chunk) {
            buf += lines[i]; buf += '\n';
            tok += tokenize(lines[i]).size();
            ++i;
        }
        chunks.push_back(Chunk{next++, doc_id, static_cast<uint32_t>(start + 1), std::move(buf)});
        if (cfg.overlap_lines && i > start + cfg.overlap_lines) i -= cfg.overlap_lines;
        else if (i == start) ++i;
    }
    return chunks;
}
} // namespace seek