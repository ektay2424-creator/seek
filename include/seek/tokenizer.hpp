#pragma once
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace seek {

inline std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> out;
    std::string cur;
    char prev = 0;
    auto flush = [&] { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };

    for (char ch : text) {
        unsigned char uc = static_cast<unsigned char>(ch);
        if (ch == '_' || !std::isalnum(uc)) { flush(); prev = ch; continue; }
        if (prev && std::islower(static_cast<unsigned char>(prev)) && std::isupper(uc)) flush();
        cur.push_back(static_cast<char>(std::tolower(uc)));
        prev = ch;
    }
    flush();
    return out;
}
} // namespace seek