// Copyright 2021, 2022 Peter Dimov.
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <pstl/glue_execution_defs.h>
#define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#define _SILENCE_CXX20_CISO646_REMOVED_WARNING

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/regex.hpp>
#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string_view>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <execution>
#include "cfoa.hpp"


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
    return !lock_.load(std::memory_order_relaxed) &&
           !lock_.exchange(true, std::memory_order_acquire);
  }

  void unlock() noexcept {
    lock_.store(false, std::memory_order_release);
  }
};


int const Th = 12; // number of threads
int const Sh = Th * Th; // number of shards

#if SIZE_MAX > UINT32_MAX
std::size_t const Reserve = 1418655 ;
//std::size_t const Reserve = 354032 ;
#else
std::size_t const Reserve = 354032 ;
#endif

using namespace std::chrono_literals;

static void print_time( std::chrono::steady_clock::time_point & t1, char const* label, std::size_t s, std::size_t size )
{
    auto t2 = std::chrono::steady_clock::now();

    std::cout << "    " << label << ": " << ( t2 - t1 ) / 1ms << " ms (s=" << s << ", size=" << size << ")" << std::endl;

    t1 = t2;
}

static std::vector<std::string> words;

static std::vector<std::string> load_preparsed_words(std::string const& filename) {
    auto is = std::ifstream(filename);
    if (!is.is_open()) {
        return {};
    }
    std::string num_words;
    std::getline(is, num_words);

    auto my_words = std::vector<std::string>();
    my_words.resize(std::stoull(num_words));
    for (auto& word : my_words) {
        std::getline(is, word);
    }
    return my_words;
}

static void save_preparsed_words(std::vector<std::string> const& words, std::string const& filename) {
    auto os = std::ofstream(filename);
    os << std::to_string(words.size()) << '\n';
    for (auto const& word : words) {
        os << word << '\n';
    }
}

static void init_words()
{
#if SIZE_MAX > UINT32_MAX

    char const* fn = "enwik9"; // http://mattmahoney.net/dc/textdata

#else

    char const* fn = "enwik8"; // ditto

#endif

    auto t1 = std::chrono::steady_clock::now();

    words = load_preparsed_words(std::string(fn) + ".words");
    if (words.empty()) {
        std::ifstream is( fn );
        std::string in( std::istreambuf_iterator<char>( is ), std::istreambuf_iterator<char>{} );

        boost::regex re( "[a-zA-Z]+");
        boost::sregex_token_iterator it( in.begin(), in.end(), re, 0 ), end;

        words.assign( it, end );

        save_preparsed_words(words, std::string(fn) + ".words");
    }

    auto t2 = std::chrono::steady_clock::now();

    std::cout << fn << ": " << words.size() << " words, " << ( t2 - t1 ) / 1ms << " ms\n\n";
}

struct ufm_single_threaded
{
    boost::unordered_flat_map<std::string_view, std::size_t> map;

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::size_t s = 0;

        for( auto const& word: words )
        {
            ++map[ word ];
            ++s;
        }

        print_time( t1, "Word count", s, map.size() );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::size_t s = 0;

        for( auto const& word: words )
        {
            std::string_view w2( word );
            w2.remove_prefix( 1 );

            s += map.contains( w2 );
        }

        print_time( t1, "Contains", s, map.size() );
    }
};

struct ufm_mutex
{
    alignas(64) boost::unordered_flat_map<std::string_view, std::size_t> map;
    alignas(64) std::mutex mtx;

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        size_t s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    auto lock = std::lock_guard( mtx );

                    ++map[ words[j] ];
                    ++s;
                }
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Word count", s, map.size() );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        size_t s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    auto lock = std::lock_guard( mtx );

                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    s += map.contains( w2 );
                }
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Contains", s, map.size() );
    }
};


struct ufm_spinlock
{
    alignas(64) boost::unordered_flat_map<std::string_view, std::size_t> map;
    alignas(64) spinlock mtx;

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        size_t s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    auto lock = std::lock_guard( mtx );

                    ++map[ words[j] ];
                    ++s;
                }
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Word count", s, map.size() );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        size_t s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    auto lock = std::lock_guard( mtx );

                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    s += map.contains( w2 );
                }
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Contains", s, map.size() );
    }
};

struct ufm_rwlock
{
    alignas(64) boost::unordered_flat_map<std::string_view, std::size_t> map;
    alignas(64) std::shared_mutex mtx;

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::lock_guard<std::shared_mutex> lock( mtx );

                    ++map[ words[j] ];
                    ++s2;
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Word count", s, map.size() );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::shared_lock<std::shared_mutex> lock(mtx);

                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    s2 += map.contains( w2 );
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Contains", s, map.size() );
    }
};

template<class Mtx> struct sync_map
{
    alignas(64) boost::unordered_flat_map<std::string_view, std::size_t> map;
    alignas(64) Mtx mtx;
};

struct ufm_sharded_mutex
{
    sync_map<std::mutex> sync[ Sh ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    auto const& word = words[ j ];

                    std::size_t hash = boost::hash<std::string_view>()( word );
                    std::size_t shard = hash % Sh;

                    std::lock_guard<std::mutex> lock( sync[ shard ].mtx );

                    ++sync[ shard ].map[ word ];
                    ++s2;
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", s, n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    std::size_t hash = boost::hash<std::string_view>()( w2 );
                    std::size_t shard = hash % Sh;

                    std::lock_guard<std::mutex> lock( sync[ shard ].mtx );

                    s2 += sync[ shard ].map.contains( w2 );
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};

//

struct prehashed
{
    std::string_view x;
    std::size_t h;

    explicit prehashed( std::string_view x_ ): x( x_ ), h( boost::hash<std::string_view>()( x_ ) ) { }

    operator std::string_view () const 
    {
        return x;
    }

    friend bool operator==( prehashed const& x, prehashed const& y ) 
    {
        return x.x == y.x;
    }

    friend bool operator==( prehashed const& x, std::string_view y ) 
    {
        return x.x == y;
    }

    friend bool operator==( std::string_view x, prehashed const& y ) 
    {
        return x == y.x;
    }
};

template<>
struct boost::hash< prehashed >
{
    using is_transparent = void;

    std::size_t operator()( prehashed const& x ) const 
    {
        return x.h;
    }

    std::size_t operator()( std::string_view x ) const 
    {
        return boost::hash<std::string_view>()( x );
    }
};

template<class Mtx> struct sync_map_prehashed
{
    alignas(64) boost::unordered_flat_map< std::string_view, std::size_t, boost::hash<prehashed>, std::equal_to<> > map;
    alignas(64) Mtx mtx;
};

template<typename Mutex>
struct ufm_sharded_mutex_prehashed
{
    sync_map_prehashed<Mutex> sync[ Sh ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view word = words[ j ];

                    prehashed x( word );
                    std::size_t shard = x.h % Sh;

                    auto lock = std::lock_guard( sync[ shard ].mtx );

                    ++sync[ shard ].map[ x ];
                    ++s2;
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", s, n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    prehashed x( w2 );
                    std::size_t shard = x.h % Sh;

                    auto lock = std::lock_guard( sync[ shard ].mtx );

                    s2 += sync[ shard ].map.contains( x );
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};

// link against tbb with -ltbb
template<typename Mutex>
struct ufm_sharded_mutex_prehashed_par
{
    sync_map_prehashed<Mutex> sync[ Sh ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::for_each(std::execution::par, words.begin(), words.end(), [&](std::string const& word) {
            prehashed x( word );
            std::size_t shard = x.h % Sh;
            auto lock = std::lock_guard( sync[ shard ].mtx );
            ++sync[ shard ].map[ x ];
        });

        std::size_t n = 0;
        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", words.size(), n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        
        auto s = std::transform_reduce(std::execution::par, words.begin(), words.end(), std::size_t{}, std::plus<>{}, [&](std::string const& word) {
            std::string_view w2( word );
            w2.remove_prefix( 1 );

            prehashed x( w2 );
            std::size_t shard = x.h % Sh;

            auto lock = std::lock_guard( sync[ shard ].mtx );

            return sync[ shard ].map.contains( x ) ? 1 : 0;
        });

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};


//

struct ufm_sharded_rwlock
{
    sync_map<std::shared_mutex> sync[ Sh ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    auto const& word = words[ j ];

                    std::size_t hash = boost::hash<std::string_view>()( word );
                    std::size_t shard = hash % Sh;

                    std::lock_guard<std::shared_mutex> lock( sync[ shard ].mtx );

                    ++sync[ shard ].map[ word ];
                    ++s2;
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", s, n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    std::size_t hash = boost::hash<std::string_view>()( w2 );
                    std::size_t shard = hash % Sh;

                    std::shared_lock<std::shared_mutex> lock( sync[ shard ].mtx );

                    s2 += sync[ shard ].map.contains( w2 );
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};

struct ufm_sharded_rwlock_prehashed
{
    sync_map_prehashed<std::shared_mutex> sync[ Sh ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view word = words[ j ];

                    prehashed x( word );
                    std::size_t shard = x.h % Sh;

                    auto lock = std::lock_guard( sync[ shard ].mtx );

                    ++sync[ shard ].map[ x ];
                    ++s2;
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", s, n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    prehashed x( w2 );
                    std::size_t shard = x.h % Sh;

                    std::shared_lock<std::shared_mutex> lock( sync[ shard ].mtx );

                    s2 += sync[ shard ].map.contains( x );
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Sh; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};

//

struct ufm_sharded_isolated
{
    struct
    {
        alignas(64) boost::unordered_flat_map<std::string_view, std::size_t> map;
    }
    sync[ Th ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, &s]{

                std::size_t s2 = 0;

                for( std::size_t j = 0; j < words.size(); ++j )
                {
                    auto const& word = words[ j ];

                    std::size_t hash = boost::hash<std::string_view>()( word );
                    std::size_t shard = hash % Th;

                    if( shard == i )
                    {
                        ++sync[ i ].map[ word ];
                        ++s2;
                    }
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Th; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", s, n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, &s]{

                std::size_t s2 = 0;

                for( std::size_t j = 0; j < words.size(); ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    std::size_t hash = boost::hash<std::string_view>()( w2 );
                    std::size_t shard = hash % Th;

                    if( shard == i )
                    {
                        s2 += sync[ i ].map.contains( w2 );
                    }
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Th; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};

struct ufm_sharded_isolated_prehashed
{
    struct
    {
        alignas(64) boost::unordered_flat_map<std::string_view, std::size_t, boost::hash<prehashed>, std::equal_to<>> map;
    }
    sync[ Th ];

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, &s]{

                std::size_t s2 = 0;

                for( std::size_t j = 0; j < words.size(); ++j )
                {
                    std::string_view word = words[ j ];

                    prehashed x( word );
                    std::size_t shard = x.h % Th;

                    if( shard == i )
                    {
                        ++sync[ i ].map[ x ];
                        ++s2;
                    }
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Th; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Word count", s, n );
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, &s]{

                std::size_t s2 = 0;

                for( std::size_t j = 0; j < words.size(); ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    prehashed x( w2 );
                    std::size_t shard = x.h % Th;

                    if( shard == i )
                    {
                        s2 += sync[ i ].map.contains( x );
                    }
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        std::size_t n = 0;

        for( std::size_t i = 0; i < Th; ++i )
        {
            n += sync[ i ].map.size();
        }

        print_time( t1, "Contains", s, n );
    }
};

template<typename Key,typename T>
struct map_policy
{
  using key_type=Key;
  using raw_key_type=typename std::remove_const<Key>::type;
  using raw_mapped_type=typename std::remove_const<T>::type;

  using init_type=std::pair<raw_key_type,raw_mapped_type>;
  using moved_type=std::pair<raw_key_type&&,raw_mapped_type&&>;
  using value_type=std::pair<const Key,T>;
  using element_type=value_type;

  static value_type& value_from(element_type& x)
  {
    return x;
  }

  template <class K,class V>
  static const raw_key_type& extract(const std::pair<K,V>& kv)
  {
    return kv.first;
  }

  static moved_type move(value_type& x)
  {
    return{
      std::move(const_cast<raw_key_type&>(x.first)),
      std::move(const_cast<raw_mapped_type&>(x.second))
    };
  }

  template<typename Allocator,typename... Args>
  static void construct(Allocator& al,element_type* p,Args&&... args)
  {
    boost::allocator_traits<Allocator>::
      construct(al,p,std::forward<Args>(args)...);
  }

  template<typename Allocator>
  static void destroy(Allocator& al,element_type* p)noexcept
  {
    boost::allocator_traits<Allocator>::destroy(al,p);
  }
};

struct ufm_concurrent_foa
{
    boost::unordered::detail::cfoa::table<
        map_policy<std::string_view, std::size_t>,
        boost::hash<std::string_view>, std::equal_to<std::string_view>,
        std::allocator<std::pair<const std::string_view,int>>> map;

    BOOST_NOINLINE void test_word_count( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    map.try_emplace(
                        []( auto& x, bool ){ ++x.second; },
                        words[j], 0 );
                    ++s2;
                }

                s += s2;
            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Word count", s, map.size() );

        std::cout << std::endl;
    }

    BOOST_NOINLINE void test_contains( std::chrono::steady_clock::time_point & t1 )
    {
        std::atomic<std::size_t> s = 0;

        std::thread th[ Th ];

        std::size_t m = words.size() / Th;

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ] = std::thread( [this, i, m, &s]{

                std::size_t s2 = 0;

                std::size_t start = i * m;
                std::size_t end = i == Th-1? words.size(): (i + 1) * m;

                for( std::size_t j = start; j < end; ++j )
                {
                    std::string_view w2( words[j] );
                    w2.remove_prefix( 1 );

                    map.find(w2, [&]( auto& ){ ++s2; } );
                }

                s += s2;

            });
        }

        for( std::size_t i = 0; i < Th; ++i )
        {
            th[ i ].join();
        }

        print_time( t1, "Contains", s, map.size() );

        std::cout << std::endl;
    }
};

//

struct record
{
    std::string label_;
    long long time_;
};

static std::vector<record> times;

template<class Map> BOOST_NOINLINE void test( int test_number, char const* label )
{
    std::cout << "#" << test_number << ": " << label << std::endl;

    Map map;

    auto t0 = std::chrono::steady_clock::now();
    auto t1 = t0;

    record rec = { label, 0 };

    map.test_word_count( t1 );
    map.test_contains( t1 );

    auto tN = std::chrono::steady_clock::now();
    std::cout << "    Total: " << ( tN - t0 ) / 1ms << " ms\n\n";

    rec.time_ = ( tN - t0 ) / 1ms;
    times.push_back( rec );
}

boost::unordered_flat_set<int> parse_args(int argc, char** argv) {
    auto s = boost::unordered_flat_set<int>();    
    for (int i=1; i<argc; ++i) {
        auto n = atoi(argv[i]);
        s.insert(n);
    }
    return s;
}

int main(int argc, char** argv)
{
    auto numbers = parse_args(argc, argv);
    init_words();

    int i=0;
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_single_threaded>( i, "boost::unordered_flat_map, single threaded" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_mutex>( i, "boost::unordered_flat_map, mutex" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_rwlock>( i, "boost::unordered_flat_map, rwlock" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_mutex>( i, "boost::unordered_flat_map, sharded mutex" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_mutex_prehashed<std::mutex>>( i, "boost::unordered_flat_map, sharded mutex, prehashed" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_mutex_prehashed<spinlock>>( i, "boost::unordered_flat_map, sharded spinlock, prehashed" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_mutex_prehashed_par<std::mutex>>( i, "boost::unordered_flat_map, sharded mutex, prehashed par" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_mutex_prehashed_par<spinlock>>( i, "boost::unordered_flat_map, sharded spinlock, prehashed par" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_rwlock>( i, "boost::unordered_flat_map, sharded rwlock" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_rwlock_prehashed>( i, "boost::unordered_flat_map, sharded rwlock, prehashed" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_isolated>( i, "boost::unordered_flat_map, sharded isolated" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_sharded_isolated_prehashed>( i, "boost::unordered_flat_map, sharded isolated, prehashed" );
    }
    if (numbers.contains(++i) || numbers.empty()) {
        test<ufm_concurrent_foa>( i, "ufm_concurrent" );
    }
    /*
    if (numbers.empty() || numbers.contains(++i)) {
        test<ufm_concurrent_foa_par>( i, "ufm_concurrent_par" );
    }
    */
    std::cout << "---\n\n";

    for( auto const& x: times )
    {
        std::cout << std::setw( 70 ) << ( x.label_ + ": " ) << std::setw( 5 ) << x.time_ << " ms\n";
    }
}
