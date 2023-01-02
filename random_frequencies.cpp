#include <boost/unordered/unordered_flat_map.hpp>

#include "cfoa.hpp"
#include "gtl/phmap.hpp"
#include "libcuckoo/cuckoohash_map.hh"
#include "tbb/concurrent_hash_map.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

static constexpr auto num_elements = 100'000'000;
static constexpr auto mask = 0xFFFFF; // 1048575
static constexpr auto lookup_key = 123;

// Runs op, measuring its runtime.
template <typename Op>
void measure(std::string_view name, int num_threads, Op op) {
    auto before = std::chrono::steady_clock::now();
    auto result = op(num_threads);
    auto after = std::chrono::steady_clock::now();
    std::cout << std::fixed << std::setprecision(3) << std::setw(10) << std::chrono::duration<double>(after - before).count()
              << " " << name << " " << num_threads << " threads (" << result << ")" << std::endl;
}

// When splitting total_work up into multiple threads, calculate amount of work per thread.
// Don't just simply divide so we get exactly the correct amount of work.
size_t calc_work(size_t total_work, int num_threads, int thread) {
    auto end = total_work * (thread + 1) / num_threads;
    auto start = total_work * thread / num_threads;
    return end - start;
}

// Creates num_threads threads that each call op(), and join() them all.
template <typename Op>
void parallel(int num_threads, Op op) {
    auto threads = std::vector<std::thread>();
    threads.reserve(num_threads);

    for (int th = 0; th < num_threads; ++th) {
        threads.emplace_back([th, op] {
            op(th);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// copied from https://rigtorp.se/spinlock/
struct spinlock {
    std::atomic<bool> lock_ = {0};

    void lock() noexcept {
        for (;;) {
            // Optimistically assume the lock is free on the first try
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lock_.load(std::memory_order_relaxed)) {
                // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
                // hyper-threads
                __builtin_ia32_pause();
            }
        }
    }

    bool try_lock() noexcept {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        lock_.store(false, std::memory_order_release);
    }
};

///////////////////////////////

// each thread creates a separate map, fills it with bounded random numbers, and finally add the count of lookup_key.
size_t doIsolated(int num_threads) {
    auto total = std::atomic<size_t>();
    parallel(num_threads, [num_threads, &total](int th) {
        auto map = boost::unordered_flat_map<size_t, size_t>();
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);
        for (size_t i = 0; i < work; ++i) {
            auto num = rng() & mask;
            ++map[num];
        }

        total += map[lookup_key];
    });
    return total.load();
}

template <typename Key, typename T>
struct map_policy {
    using key_type = Key;
    using raw_key_type = typename std::remove_const<Key>::type;
    using raw_mapped_type = typename std::remove_const<T>::type;

    using init_type = std::pair<raw_key_type, raw_mapped_type>;
    using moved_type = std::pair<raw_key_type&&, raw_mapped_type&&>;
    using value_type = std::pair<const Key, T>;
    using element_type = value_type;

    static value_type& value_from(element_type& x) {
        return x;
    }

    template <class K, class V>
    static const raw_key_type& extract(const std::pair<K, V>& kv) {
        return kv.first;
    }

    static moved_type move(value_type& x) {
        return {std::move(const_cast<raw_key_type&>(x.first)), std::move(const_cast<raw_mapped_type&>(x.second))};
    }

    template <typename Allocator, typename... Args>
    static void construct(Allocator& al, element_type* p, Args&&... args) {
        boost::allocator_traits<Allocator>::construct(al, p, std::forward<Args>(args)...);
    }

    template <typename Allocator>
    static void destroy(Allocator& al, element_type* p) noexcept {
        boost::allocator_traits<Allocator>::destroy(al, p);
    }
};

size_t doCfoa(int num_threads) {
    auto map = boost::unordered::detail::cfoa::table<map_policy<size_t, size_t>,
                                                     boost::hash<size_t>,
                                                     std::equal_to<size_t>,
                                                     std::allocator<std::pair<const size_t, size_t>>>();

    parallel(num_threads, [&map, num_threads](int th) {
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);

        // a bunch of inserts
        for (size_t i = 0; i < work; ++i) {
            auto num = rng() & mask;
            map.try_emplace(
                [](auto& x, bool) {
                    ++x.second;
                },
                num,
                0);
        }
    });

    size_t r = 0;
    map.find(lookup_key, [&](auto const& x) {
        r = x.second;
    });
    return r;
}

size_t doCuckooHash(int num_threads) {
    auto map = libcuckoo::cuckoohash_map<size_t, size_t>();
    parallel(num_threads, [&map, num_threads](int th) {
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);
        for (size_t i = 0; i < work; ++i) {
            auto num = rng() & mask;
            // see https://github.com/efficient/libcuckoo/blob/master/examples/count_freq.cc#L32
            // If the number is already in the table, it will increment
            // its count by one. Otherwise it will insert a new entry in
            // the table with count one.
            map.upsert(
                num,
                [](size_t& n) {
                    ++n;
                },
                1);
        }
    });

    size_t r = 0;
    map.find_fn(lookup_key, [&](auto const& x) {
        r = x;
    });
    return r;
}

size_t doGtl(int num_threads) {
    auto map = gtl::parallel_flat_hash_map<size_t,
                                           size_t,
                                           gtl::priv::hash_default_hash<size_t>,
                                           gtl::priv::hash_default_eq<size_t>,
                                           std::allocator<std::pair<const size_t, size_t>>,
                                           4,
                                           spinlock>();

    parallel(num_threads, [&map, num_threads](int th) {
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);

        // a bunch of inserts
        for (size_t i = 0; i < work; ++i) {
            auto num = rng() & mask;
            map.lazy_emplace_l(
                num,
                [](auto& v) {
                    ++v.second;
                },
                [num](auto const& ctor) {
                    ctor(num, 1);
                });
        }
    });

    size_t r = 0;
    map.if_contains(lookup_key, [&](auto const& x) {
        r = x.second;
    });
    return r;
}

size_t doTbb(int num_threads) {
    using map_t = tbb::concurrent_hash_map<size_t, size_t>;
    auto map = map_t();
    parallel(num_threads, [&map, num_threads](int th) {
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);

        for (size_t i = 0; i < work; ++i) {
            auto num = rng() & mask;

            auto acc = map_t::accessor();
            map.emplace(acc, num, 0);
            ++acc->second;
        }
    });

    auto acc = map_t::const_accessor();
    map.find(acc, lookup_key);
    return acc->second;
}

int main(int argc, char** argv) {
    measure("tbb::concurrent_hash_map", 1, doTbb);
    measure("tbb::concurrent_hash_map", std::thread::hardware_concurrency(), doTbb);
    measure("boost::unordered_flat_map isolated", 1, doIsolated);
    measure("boost::unordered_flat_map isolated", std::thread::hardware_concurrency(), doIsolated);
    measure("boost::unordered::detail::cfoa::table", 1, doCfoa);
    measure("boost::unordered::detail::cfoa::table", std::thread::hardware_concurrency(), doCfoa);
    measure("libcuckoo::cuckoohash_map", 1, doCuckooHash);
    measure("libcuckoo::cuckoohash_map", std::thread::hardware_concurrency(), doCuckooHash);
    measure("gtl::parallel_flat_hash_map", 1, doGtl);
    measure("gtl::parallel_flat_hash_map", std::thread::hardware_concurrency(), doGtl);
}
