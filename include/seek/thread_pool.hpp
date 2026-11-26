#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace seek {

class ThreadPool {
public:
    explicit ThreadPool(size_t n = std::thread::hardware_concurrency()) {
        if (n == 0) n = 1;
        for (size_t i = 0; i < n; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~ThreadPool() { stop(); }

    void enqueue(std::function<void()> job) {
        { std::lock_guard<std::mutex> lk(mu_); jobs_.push(std::move(job)); }
        cv_.notify_one();
    }

    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        idle_cv_.wait(lk, [this] { return jobs_.empty() && active_ == 0; });
    }

    void stop() {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
    }

private:
    void loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) return;
                job = std::move(jobs_.front()); jobs_.pop();
                ++active_;
            }
            job();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--active_ == 0 && jobs_.empty()) idle_cv_.notify_all();
            }
        }
    }
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> jobs_;
    std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    size_t active_ = 0;
    bool stop_ = false;
};
} // namespace seek