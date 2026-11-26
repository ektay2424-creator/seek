#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace seek {

class Trie {
public:
    void insert(std::string_view w) {
        Node* cur = &root_;
        for (char c : w) {
            auto& nxt = cur->children[c];
            if (!nxt) nxt = std::make_unique<Node>();
            cur = nxt.get();
        }
        cur->terminal = true;
    }

    std::vector<std::string> complete(std::string_view prefix, size_t limit = 10) const {
        const Node* cur = &root_;
        for (char c : prefix) {
            auto it = cur->children.find(c);
            if (it == cur->children.end()) return {};
            cur = it->second.get();
        }
        std::vector<std::string> out;
        std::string buf(prefix);
        dfs(cur, buf, out, limit);
        return out;
    }

private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> children;
        bool terminal = false;
    };
    static void dfs(const Node* n, std::string& buf, std::vector<std::string>& out, size_t limit) {
        if (out.size() >= limit) return;
        if (n->terminal) out.push_back(buf);
        for (auto& [c, child] : n->children) {
            buf.push_back(c);
            dfs(child.get(), buf, out, limit);
            buf.pop_back();
            if (out.size() >= limit) return;
        }
    }
    Node root_;
};
} // namespace seek