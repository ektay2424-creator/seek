# Seek: Local-First Hybrid File & Code Search Engine

Seek is a lightweight, single-binary C++17 command-line search engine that indexes local directories (`.pdf`, `.md`, `.txt`, `.cpp`, `.py`, `.js`) completely offline. It answers both exact-symbol and natural-language queries in milliseconds by fusing **BM25 lexical scoring** with **BGE-small neural embeddings** via **Reciprocal Rank Fusion**.

## 🚀 Demo

```text
$ seek query "concurrent workers consuming jobs" --embedder onnx
[0.0327869] include\seek\thread_pool.hpp:1
    #pragma once  #include <atomic>  #include <condition_variable>  #include <functional> ...

 Features
Multithreaded Ingestion: Crawls, chunks, and indexes 10,000+ files at 2,173 files/sec.

Hybrid Retrieval: Inverted Index (Okapi BM25) + Vector Store (Cosine Similarity) fused with RRF.

Zero Python Overhead: Runs ONNX Runtime natively in C++ with a custom WordPiece tokenizer.

Incremental Updates: seek watch detects filesystem changes and re-indexes only modified files.

Fast: 8.96 ms p95 query latency across 18,774 chunks, at just 49.7 MB peak RSS.

🛠️ Build Instructions
Windows (MinGW)
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

Linux / macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

🧠 Architecture
Lexical Index: Custom std::unordered_map postings list with Okapi BM25.

Prefix Autocomplete: Trie with O(L) lookup.

Vector Store: Contiguous flat float32 matrix with O(N log K) Min-Heap top-K extraction.

Rank Fusion: Reciprocal Rank Fusion (1 / (k + rank)) to merge lexical and semantic rankings without score normalization.

Concurrency: Lock-safe producer-consumer queue using std::condition_variable.

🧪 Testing
Runs 62 automated tests covering tokenization, BM25 rank order, RRF convexity, thread-pool correctness, and serialization.
ctest --test-dir build --output-on-failure

📄 License
MIT

Save that. Once CI turns **green**, run:
```cmd
git add README.md
git commit -m "docs: add comprehensive README with benchmarks and architecture"
git push origin main