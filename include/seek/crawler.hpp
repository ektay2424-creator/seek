#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace seek {

inline bool indexable(const std::filesystem::path& p) {
    static const char* exts[] = {".txt",".md",".cpp",".hpp",".h",".c",".py",".js",".json"};
    auto e = p.extension().string();
    for (auto* x : exts) if (e == x) return true;
    return false;
}

inline bool ignored(const std::filesystem::path& p) {
    auto n = p.filename().string();
    return n == ".git" || n == "node_modules" || n == "build" || n == ".seekidx";
}

inline std::vector<std::filesystem::path> crawl(const std::vector<std::string>& roots) {
    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    for (auto& root : roots) {
        std::error_code ec;
        if (!fs::exists(root, ec)) continue;
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
        for (; it != end; it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            const auto& p = it->path();
            if (it->is_directory(ec)) { if (ignored(p)) it.disable_recursion_pending(); continue; }
            if (it->is_regular_file(ec) && indexable(p)) files.push_back(p);
        }
    }
    return files;
}
} // namespace seek