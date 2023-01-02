#include <boost/unordered/unordered_flat_map.hpp>

#include "cfoa.hpp"
#include "cuckoohash_map.hh"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

static constexpr auto num_elements = 100'000'000;
static constexpr auto mask = 0xFFFFF; // 1048575

template <typename Op> void measure(std::string_view name, Op op) {
  auto before = std::chrono::steady_clock::now();
  auto result = op();
  auto after = std::chrono::steady_clock::now();
  std::cout << std::setprecision(4)
            << std::chrono::duration<double>(after - before).count() << " "
            << name << " (" << result << ")" << std::endl;
}

void benchSingleThreaded() {
  measure("single threaded boost::unordered_flat_map", []() {
    auto rng = ankerl::nanobench::Rng();
    auto map = boost::unordered_flat_map<size_t, size_t>();

    // a bunch of inserts
    for (size_t i = 0; i < num_elements; ++i) {
      ++map[rng() & mask];
    }

    // sum val
    auto sum = size_t();
    for (auto const &[key, val] : map) {
      sum += val;
    }
    return sum;
  });
}

void benchCuckooHash(int num_threads) {
  measure("single threaded libcuckoo::cuckoohash_map", [num_threads]() {
    auto threads = std::vector<std::thread>();

    auto map = libcuckoo::cuckoohash_map<size_t, size_t>();
    for (int th = 0; th < num_threads; ++th) {
      auto work = num_elements / num_threads;
      if (th == num_threads - 1) {
        // calculate remainder to make sure we do exactly the same work as
        // everybody else
        work = num_elements - (num_threads - 1) * work;
      }
      threads.emplace_back([th, work, &map] {
        auto rng = ankerl::nanobench::Rng(th);

        // a bunch of inserts
        for (size_t i = 0; i < work; ++i) {
          auto num = rng() & mask;
          // see
          // https://github.com/efficient/libcuckoo/blob/master/examples/count_freq.cc#L32
          // If the number is already in the table, it will increment
          // its count by one. Otherwise it will insert a new entry in
          // the table with count one.
          map.upsert(
              num, [](size_t &n) { ++n; }, 1);
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }

    // sum val
    auto sum = size_t();
    auto lt = map.lock_table();
    for (auto const &it : lt) {
      sum += it.second;
    }
    return sum;
  });
}

int main(int argc, char **argv) {
  benchSingleThreaded();
  benchCuckooHash(1);
  benchCuckooHash(std::thread::hardware_concurrency());
}
