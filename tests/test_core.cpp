// tests/test_core.cpp
#include "seek/tokenizer.hpp"
#include "seek/chunker.hpp"
#include "seek/inverted_index.hpp"
#include "seek/trie.hpp"
#include "seek/vector_index.hpp"
#include "seek/thread_pool.hpp"
#include "seek/index_store.hpp"
#include "seek/embedder.hpp"

#include <atomic>
#include <cstdio>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>

using namespace seek;

// ---- minimal test harness ----
static int g_fail = 0, g_pass = 0;
#define CHECK(cond) do { \
    if (!(cond)) { std::printf("FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
    else { ++g_pass; } \
} while (0)
#define TEST(name) static void test_##name()

// ---- 1. tokenizer ----
TEST(tokenizer_basic) {
    auto t = tokenize("the quick brown fox");
    CHECK(t.size() == 4);
    CHECK(t[0] == "the"); CHECK(t[3] == "fox");
}

TEST(tokenizer_camel_and_punct) {
    auto t = tokenize("getUserId_byId(x,");
    // expect: get, user, id, by, id, x
    CHECK(t.size() == 6);
    CHECK(t[0] == "get"); CHECK(t[1] == "user"); CHECK(t[2] == "id");
    CHECK(t[3] == "by");  CHECK(t[4] == "id");   CHECK(t[5] == "x");
}

TEST(tokenizer_empty) {
    CHECK(tokenize("").empty());
    CHECK(tokenize("!!! ,,, ___ ").empty());
}

// ---- 2. chunker ----
TEST(chunker_splits_and_records_lines) {
    std::string text;
    for (int i = 0; i < 600; ++i) text += "line " + std::to_string(i) + "\n";
    auto chunks = chunk_document(0, text, {100, 2});   // 100 tokens per chunk
    CHECK(chunks.size() >= 2);
    CHECK(chunks[0].start_line == 1);                  // 1-based
    CHECK(chunks[1].start_line > chunks[0].start_line);
    for (auto& c : chunks) CHECK(c.doc_id == 0);
}

// ---- 3. BM25 ranks exact match first ----
TEST(bm25_short_chunk_wins) {
    InvertedIndex idx;
    idx.add_chunk(0, "the quick brown fox");              // 4 tokens
    idx.add_chunk(1, "the lazy brown dog jumps over the fox"); // 8 tokens
    idx.add_chunk(2, "completely unrelated content here");

    auto terms = tokenize("fox");
    auto hits = idx.search(terms, 5);
    CHECK(hits.size() == 2);
    CHECK(hits[0].id == 0);        // shorter chunk, same tf, higher BM25
    CHECK(hits[1].id == 1);
}

TEST(bm25_no_match_returns_empty) {
    InvertedIndex idx;
    idx.add_chunk(0, "hello world");
    auto hits = idx.search(tokenize("zzz"), 5);
    CHECK(hits.empty());
}

// ---- 4. trie prefix completion ----
TEST(trie_completion) {
    Trie trie;
    for (auto& w : {"char", "chars", "chunk", "chunks", "chunk_id", "dog"})
        trie.insert(w);

    auto c = trie.complete("cha", 10);
    CHECK(c.size() == 2);
    // both must start with "cha"
    for (auto& s : c) CHECK(s.rfind("cha", 0) == 0);

    auto cu = trie.complete("chu", 10);
    CHECK(cu.size() == 3);

    CHECK(trie.complete("xyz", 10).empty());
}

// ---- 5. RRF fusion ----
TEST(rrf_fusion_agrees_on_top) {
    // chunk 0 is #1 in lex only; chunk 2 is #1 in vec only;
    // chunk 1 is #2 in BOTH lists -> the "agreement" winner.
    std::vector<ScoredChunk> lex = {{0, 10}, {1, 8}};
    std::vector<ScoredChunk> vec = {{2, 0.9f}, {1, 0.8f}};

    auto fused = rrf_fuse({lex, vec}, 3);
    CHECK(fused.size() == 3);
    CHECK(fused[0].id == 1);
}

TEST(rrf_top1_beats_middle_consistency) {
    // Convexity check: #1 + #3 beats #2 + #2 when both lists are 3 long.
    std::vector<ScoredChunk> a = {{0, 10}, {1, 8}, {2, 3}};
    std::vector<ScoredChunk> b = {{2, 0.9f}, {1, 0.8f}, {0, 0.5f}};

    auto fused = rrf_fuse({a, b}, 3);
    CHECK(fused.size() == 3);
    CHECK(fused.back().id == 1);   // #2-in-both loses to #1-in-one
}

// ---- 6. thread pool runs every job ----
TEST(thread_pool_runs_all) {
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4);
        for (int i = 0; i < 100; ++i) pool.enqueue([&] { counter.fetch_add(1); });
        pool.wait_idle();
    }
    CHECK(counter.load() == 100);
}

// ---- 7. save / load roundtrip ----
TEST(serialize_roundtrip) {
    SeekIndex idx;
    idx.docs.push_back({0, "a.txt", 111});
    idx.chunks.push_back({0, 0, 1, "hello world"});
    idx.chunks.push_back({1, 0, 2, "second chunk"});
    idx.inv.add_chunk(0, "hello world");
    idx.inv.add_chunk(1, "second chunk");
    for (auto& t : tokenize("hello world")) idx.trie.insert(t);
    for (auto& t : tokenize("second chunk")) idx.trie.insert(t);
    idx.vec.add(0, {1.0f, 0.0f, 0.0f, 0.0f});
    idx.vec.add(1, {0.0f, 1.0f, 0.0f, 0.0f});

    std::string path = "seek_test_roundtrip.seekidx";
    CHECK(save_index(idx, path));

    SeekIndex loaded;
    CHECK(load_index(loaded, path));
    CHECK(loaded.docs.size() == 1);
    CHECK(loaded.chunks.size() == 2);
    CHECK(loaded.docs[0].path == "a.txt");
    CHECK(loaded.chunks[1].text == "second chunk");
    CHECK(loaded.inv.num_chunks() == 2);
    CHECK(loaded.vec.size() == 2);

    // trie was rebuilt from the inverted index
    auto c = loaded.trie.complete("hel", 5);
    CHECK(c.size() == 1 && c[0] == "hello");

    std::filesystem::remove(path);
}



TEST(hash_embedder_deterministic_and_normalized) {
    HashEmbedder e(64);
    auto a = e.embed("dynamic programming knapsack");
    auto b = e.embed("dynamic programming knapsack");
    CHECK(a.size() == 64);
    CHECK(a == b);                                // deterministic

    float n2 = 0; for (float x : a) n2 += x * x;
    CHECK(std::abs(std::sqrt(n2) - 1.0f) < 1e-4f); // unit norm
}

TEST(hybrid_rrf_returns_real_chunks) {
    SeekIndex idx;
    HashEmbedder emb(64);

    std::vector<std::string> texts = {
        "the quick brown fox jumps over the lazy dog",
        "dynamic programming 0/1 knapsack memoized table",
        "bst lowest common ancestor binary search tree"
    };
    for (size_t i = 0; i < texts.size(); ++i) {
        idx.chunks.push_back({(ChunkId)i, 0, 1, texts[i]});
        idx.inv.add_chunk((ChunkId)i, texts[i]);
        idx.vec.add((ChunkId)i, emb.embed(texts[i]));
    }
    idx.docs.push_back({0, "test.txt", 0});

    auto qv = emb.embed("knapsack");
    auto lex = idx.inv.search(tokenize("knapsack"), 3);
    auto sem = idx.vec.search(qv, 3);
    auto fused = rrf_fuse({lex, sem}, 3);

    CHECK(!fused.empty());
    // chunk 1 mentions knapsack, so lexical ranks it #1 -> it must be in the top-K
    bool found = false;
    for (auto& h : fused) if (h.id == 1) found = true;
    CHECK(found);
}

// ---- runner ----
int main() {
    test_tokenizer_basic();
    test_tokenizer_camel_and_punct();
    test_tokenizer_empty();
    test_chunker_splits_and_records_lines();
    test_bm25_short_chunk_wins();
    test_bm25_no_match_returns_empty();
    test_trie_completion();
    test_rrf_fusion_agrees_on_top();
    test_thread_pool_runs_all();
    test_serialize_roundtrip();
    test_hash_embedder_deterministic_and_normalized();
    test_hybrid_rrf_returns_real_chunks();

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}