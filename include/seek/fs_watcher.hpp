#pragma once
#include "crawler.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace seek {

struct FileStamp {
    std::filesystem::file_time_type mtime;
    uintmax_t size = 0;
    bool operator==(const FileStamp& o) const { return mtime == o.mtime && size == o.size; }
};

class PollingWatcher {
public:
    explicit PollingWatcher(std::vector<std::string> roots) : roots_(std::move(roots)) {}

    // Returns paths that were added, modified, or deleted since last scan().
    std::vector<std::string> scan() {
        namespace fs = std::filesystem;
        std::vector<std::string> changed;
        std::unordered_map<std::string, FileStamp> current;

        for (auto& path : crawl(roots_)) {
            std::error_code ec;
            auto t  = fs::last_write_time(path, ec); if (ec) continue;
            auto sz = fs::file_size(path, ec);       if (ec) continue;
            std::string key = path.string();
            FileStamp s{t, sz};
            current[key] = s;
            auto it = seen_.find(key);
            if (it == seen_.end() || !(it->second == s)) changed.push_back(key);
        }
        for (auto& kv : seen_)
            if (!current.count(kv.first)) changed.push_back(kv.first);

        seen_ = std::move(current);
        return changed;
    }

private:
    std::vector<std::string> roots_;
    std::unordered_map<std::string, FileStamp> seen_;
};
} // namespace seek