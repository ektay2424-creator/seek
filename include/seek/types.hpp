#pragma once
#include <cstdint>
#include <string>

namespace seek {
using ChunkId = uint32_t;
using DocId   = uint32_t;

struct Chunk {
    ChunkId  id         = 0;
    DocId    doc_id     = 0;
    uint32_t start_line = 0;
    std::string text;
};

struct Document {
    DocId    id = 0;
    std::string path;
    uint64_t content_hash = 0;
};

struct ScoredChunk { ChunkId id; float score; };
} // namespace seek