#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace seek {

class WordPieceTokenizer {
public:
    explicit WordPieceTokenizer(const std::string& vocab_path, size_t max_len = 512)
        : max_len_(max_len) {
        std::ifstream f(vocab_path);
        if (!f) throw std::runtime_error("Cannot open vocab: " + vocab_path);
        std::string line;
        int idx = 0;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();
            vocab_[line] = idx++;
        }
        unk_id_ = get_id("[UNK]");
        cls_id_ = get_id("[CLS]");
        sep_id_ = get_id("[SEP]");
        if (unk_id_ < 0 || cls_id_ < 0 || sep_id_ < 0)
            throw std::runtime_error("vocab.txt missing [UNK]/[CLS]/[SEP]");
    }

    // Returns {input_ids, attention_mask}; both have length ≤ max_len_
    std::pair<std::vector<int64_t>, std::vector<int64_t>>
    encode(const std::string& text) const {
        std::vector<int64_t> ids;
        ids.push_back(cls_id_);
        for (auto& w : basic_split(text)) {
            for (auto id : wordpiece(w)) {
                if (ids.size() + 1 >= max_len_) goto done;
                ids.push_back(id);
            }
        }
    done:
        ids.push_back(sep_id_);
        return {ids, std::vector<int64_t>(ids.size(), 1)};
    }

private:
    int get_id(const std::string& t) const {
        auto it = vocab_.find(t);
        return it == vocab_.end() ? -1 : it->second;
    }

    static std::vector<std::string> basic_split(const std::string& text) {
        std::vector<std::string> out;
        std::string cur;
        auto flush = [&] { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
        for (unsigned char ch : text) {
            if (std::isspace(ch)) { flush(); continue; }
            if (std::ispunct(ch)) {
                flush();
                out.push_back(std::string(1, (char)ch));
                continue;
            }
            cur.push_back((char)std::tolower(ch));
        }
        flush();
        return out;
    }

    std::vector<int64_t> wordpiece(const std::string& word) const {
        std::vector<int64_t> ids;
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int64_t found = -1;
            while (start < end) {
                std::string sub = word.substr(start, end - start);
                if (start > 0) sub = "##" + sub;
                auto it = vocab_.find(sub);
                if (it != vocab_.end()) { found = it->second; break; }
                --end;
            }
            if (found < 0) { ids.push_back(unk_id_); return ids; }
            ids.push_back(found);
            start = end;
        }
        return ids;
    }

    std::unordered_map<std::string, int> vocab_;
    int unk_id_ = -1, cls_id_ = -1, sep_id_ = -1;
    size_t max_len_;
};
} // namespace seek