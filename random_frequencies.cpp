#include <boost/thread/shared_mutex.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <tbb/spin_rw_mutex.h>

#include "cfoa.hpp"
#include "folly/RWSpinLock.h"
#include "gtl/phmap.hpp"
#include "libcuckoo/cuckoohash_map.hh"
#include "oneapi/tbb/concurrent_hash_map.h"
#include "oneapi/tbb/spin_rw_mutex.h"
#include "rw_spinlock.hpp"

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
static constexpr auto lookup_key = 0;
static constexpr auto num_benchmark_runs_for_median = 9;

// uniformly distributed number in range [0, mask]
#if 0

inline size_t calc_index(ankerl::nanobench::Rng& rng) {
    return rng() & mask;
}

#else

// biased numbers in range
inline size_t calc_index(ankerl::nanobench::Rng& rng) {
    auto n = rng();
    auto shift = (n & 15) + 48;
    return n >> shift;
}

#endif

// Runs op, measuring its runtime.
template <typename Op>
void measure(std::string_view name, int num_threads, Op op) {
    if (0 == num_threads) {
        std::cout << name << "; ";
        return;
    }

    auto durations = std::vector<double>(num_benchmark_runs_for_median);
    for (auto& dur : durations) {
        auto before = std::chrono::steady_clock::now();
        auto result = op(num_threads);
        auto after = std::chrono::steady_clock::now();
        if (0 == result) {
            throw std::runtime_error("somethig wrong in " + std::string(name));
        }
        // std::cout << "(r=" << result << ")";
        dur = std::chrono::duration<double>(after - before).count();
    }
    std::sort(durations.begin(), durations.end());
    auto mid1 = (durations.size() - 1) / 2;
    auto mid2 = (durations.size()) / 2;
    auto t = (durations[mid1] + durations[mid2]) / 2;
    auto million_elements_per_second = num_elements / (t * 1e6);
    std::cout << million_elements_per_second << ";";
    std::cout.flush();
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
            ++map[calc_index(rng)];
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

template <typename Mtx>
size_t doCfoa(int num_threads) {
    auto map = boost::unordered::detail::cfoa::table<map_policy<size_t, size_t>,
                                                     boost::hash<size_t>,
                                                     std::equal_to<size_t>,
                                                     std::allocator<std::pair<const size_t, size_t>>,
                                                     Mtx>();

    parallel(num_threads, [&map, num_threads](int th) {
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);

        // a bunch of inserts
        for (size_t i = 0; i < work; ++i) {
            map.try_emplace(
                [](auto& x, bool) {
                    ++x.second;
                },
                calc_index(rng),
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
            // see https://github.com/efficient/libcuckoo/blob/master/examples/count_freq.cc#L32
            // If the number is already in the table, it will increment
            // its count by one. Otherwise it will insert a new entry in
            // the table with count one.
            map.upsert(
                calc_index(rng),
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

template <size_t NumSubmaps>
size_t doGtl(int num_threads) {
    auto map = gtl::parallel_flat_hash_map<size_t,
                                           size_t,
                                           gtl::priv::hash_default_hash<size_t>,
                                           gtl::priv::hash_default_eq<size_t>,
                                           std::allocator<std::pair<const size_t, size_t>>,
                                           NumSubmaps,
                                           spinlock>();

    parallel(num_threads, [&map, num_threads](int th) {
        auto work = calc_work(num_elements, num_threads, th);
        auto rng = ankerl::nanobench::Rng(th);

        // a bunch of inserts
        for (size_t i = 0; i < work; ++i) {
            auto num = calc_index(rng);
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
            auto acc = map_t::accessor();
            map.emplace(acc, calc_index(rng), 0);
            ++acc->second;
        }
    });

    auto acc = map_t::const_accessor();
    map.find(acc, lookup_key);
    return acc->second;
}

int main(int argc, char** argv) {
    for (int i = 0; i <= 20; ++i) {
        std::cout << i << ";";
        //measure("boost::unordered_flat_map isolated", i, doIsolated);
        measure("cfoa<rw_spinlock>", i, doCfoa<rw_spinlock>);
        //measure("cfoa<std::shared_mutex>", i, doCfoa<std::shared_mutex>);
        //measure("cfoa<boost::shared_mutex>", i, doCfoa<boost::shared_mutex>);
        //measure("cfoa<folly::RWSpinLock>", i, doCfoa<folly::RWSpinLock>);
        //measure("cfoa<folly::RWTicketSpinLockT<32>>", i, doCfoa<folly::RWTicketSpinLockT<32>>);
        //measure("cfoa<tbb::spin_rw_mutex>", i, doCfoa<tbb::spin_rw_mutex>);
        //measure("libcuckoo::cuckoohash_map", i, doCuckooHash);
        //measure("gtl::parallel_flat_hash_map<4>", i, doGtl<4>);
        //measure("gtl::parallel_flat_hash_map<6>", i, doGtl<6>);
        //measure("tbb::concurrent_hash_map", i, doTbb);
        std::cout << std::endl;
    }
}
