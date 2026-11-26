// benchmarks/bench.cpp
#include "seek/crawler.hpp"
#include "seek/embedder.hpp"
#include "seek/index_store.hpp"
#include "seek/indexer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
#endif

using namespace seek;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static double rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0.0;
    return double(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0)
            return std::stod(line.substr(6)) / 1024.0;
    return 0.0;
#endif
}

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size() - 1, (size_t)(p * v.size()));
    return v[i];
}

static void print_table(const std::string& label, size_t n, double ms, double rss) {
    std::printf("%-10s %8zu  %10.3f ms  %8.1f MB\n", label.c_str(), n, ms, rss);
}

// bench index <dir...>
static int bench_index(const std::vector<std::string>& roots) {
    auto files = crawl(roots);
    std::printf("corpus: %zu files\n", files.size());

    auto embedder = make_embedder("hash", 384);
    auto t0 = Clock::now();
    auto idx = build_index(roots, *embedder);
    auto t1 = Clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fps = files.empty() ? 0.0 : (files.size() / (ms / 1000.0));
    std::printf("%-10s %8zu  %10.3f ms  %8.1f MB   (%.1f files/sec)\n",
                "index", idx.chunks.size(), ms, rss_mb(), fps);
    std::printf("indexed %zu docs, %zu chunks, vocab %zu\n",
                idx.docs.size(), idx.chunks.size(), idx.inv.vocab_size());

    if (!save_index(idx, ".seekidx.bench")) {
        std::printf("save failed\n"); return 1;
    }
    std::printf("saved -> .seekidx.bench\n");
    return 0;
}

// bench query <index> [n_queries]
static int bench_query(const std::string& index_path, int n_queries) {
    SeekIndex idx;
    if (!load_index(idx, index_path)) { std::printf("load failed\n"); return 1; }

    auto embedder = make_embedder("hash", 384);
    static const char* kQueries[] = {
        "inverted index postings", "binary serialization term frequency",
        "thread safe producer consumer", "trie prefix autocomplete",
        "min heap top k", "reciprocal rank fusion", "vector cosine similarity",
        "thread pool condition variable", "sha 256 content hash",
        "lock free embedding cache"
    };
    const size_t kQ = sizeof(kQueries) / sizeof(kQueries[0]);

    std::vector<double> lat_ms;
    lat_ms.reserve(n_queries);
    for (int i = 0; i < n_queries; ++i) {
        const char* q = kQueries[i % kQ];
        auto t0 = Clock::now();
        auto terms   = tokenize(q);
        auto lexical = idx.inv.search(terms, 20);
        auto qvec    = embedder->embed(q);
        auto semantic= idx.vec.search(qvec, 20);
        auto hits    = rrf_fuse({lexical, semantic}, 5);
        (void)hits;
        auto t1 = Clock::now();
        lat_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::printf("queries: %d over %zu docs, %zu chunks\n",
                n_queries, idx.docs.size(), idx.chunks.size());
    std::printf("p50 = %.3f ms   p95 = %.3f ms   p99 = %.3f ms\n",
                pct(lat_ms, 0.50), pct(lat_ms, 0.95), pct(lat_ms, 0.99));
    return 0;
}

// bench synth <n>  — creates N small files in a temp dir, then times indexing
static int bench_synth(int n) {
    auto dir = fs::temp_directory_path() / "seek_synth";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    std::ofstream(dummy); // no-op to satisfy unused warnings on some compilers
    for (int i = 0; i < n; ++i) {
        std::ofstream out(dir / ("f" + std::to_string(i) + ".md"));
        out << "doc " << i << " dynamic programming 0/1 knapsack memoized table\n"
            << "binary search tree lowest common ancestor prefix trie min heap\n"
            << "inverted index bm25 term frequency inverse document frequency\n";
    }

    auto t0 = Clock::now();
    auto embedder = make_embedder("hash", 384);
    auto idx = build_index({dir.string()}, *embedder);
    auto t1 = Clock::now();

    double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fps = n / (ms / 1000.0);
    std::printf("%-10s %8zu  %10.3f ms  %8.1f MB   (%.1f files/sec)\n",
                "synth", idx.chunks.size(), ms, rss_mb(), fps);
    fs::remove_all(dir, ec);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage:\n"
                    "  seek_bench index <dir...>\n"
                    "  seek_bench query <index> [n_queries]\n"
                    "  seek_bench synth <n_files>\n");
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "index") {
        std::vector<std::string> roots(argv + 2, argv + argc);
        return bench_index(roots);
    }
    if (cmd == "query") {
        int n = (argc >= 4) ? std::stoi(argv[3]) : 1000;
        return bench_query(argv[2], n);
    }
    if (cmd == "synth") return bench_synth(std::stoi(argv[2]));

    std::printf("unknown bench command: %s\n", cmd.c_str());
    return 1;
}