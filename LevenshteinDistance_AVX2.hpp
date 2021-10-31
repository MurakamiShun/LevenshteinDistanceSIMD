/*
 * LevenshteinDistans_AVX2: https://github.com/MurakamiShun/LevenshteinDistance_AVX2
 * Copyright (c) 2021 Murakami Shun
 * 
 * Released under the MIT Lisence.
 */
#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <tuple>

#ifdef __SSE4_1__
#include <nmmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/*
uint64_t levenshtein_distance(const std::string_view str1, const std::string_view str2){
    std::vector<uint64_t> dist((str1.size() + 1) * (str2.size() + 1), 0);
    const auto cord_to_idx = [w = str1.size() + 1](std::size_t x, std::size_t y){
        return w*y + x;
    };
    for(std::size_t x = 1; x <= str1.size(); ++x) dist[cord_to_idx(x, 0)] = x;
    for(std::size_t y = 1; y <= str2.size(); ++y) dist[cord_to_idx(0, y)] = y;
    
    for(std::size_t y = 1; y <= str2.size(); ++y){
        for(std::size_t x = 1; x <= str1.size(); ++x){
            dist[cord_to_idx(x, y)] = std::min(std::min(
                dist[cord_to_idx(x - 1, y)]     + 1,
                dist[cord_to_idx(x,     y - 1)] + 1),
                dist[cord_to_idx(x - 1, y - 1)] + (str1[x-1] == str2[y-1] ? 0 : 1)
            );
        }
    }
    // Debug code
    #ifdef ENABLE_DEBUG
    std::cout << "      ";
    for(std::size_t x = 0; x < str1.size(); ++x) std::cout << str1[x] << "  ";
    std::cout << std::endl;
    for(std::size_t y = 0; y <= str2.size(); ++y){
        std::cout << (y != 0 ? str2[y-1] : ' ') << " ";
        for(std::size_t x = 0; x <= str1.size(); ++x){
            std::cout << std::setw(2) << dist[cord_to_idx(x, y)] << " ";
        }
        std::cout << std::endl;
    }
    #endif

    return dist[cord_to_idx(str1.size(), str2.size())];
}
*/
namespace LevenshteinDistansAVX2{
template<typename Container>
uint32_t levenshtein_distance_nosimd(const Container& str1, const Container& str2){
    const auto [short_str, long_str] = (str1.size() < str2.size() ? std::tie(str1,str2) : std::tie(str2, str1));
    struct Dist{
        std::vector<uint32_t> dist;
        const std::size_t n;
        Dist(const Container& str):
        dist(str.size() * 2, 0),
        n(str.size()){
            for(std::size_t x = 0; x < n; ++x) dist[x] = x + 1;
        }
        void set(std::size_t x, std::size_t y, uint32_t val) {
            dist[x - 1 + (y%2)*n] = val;
        }
        uint32_t operator()(std::size_t x, std::size_t y) const {
            if(x == 0) return y;
            else return dist[x - 1 + (y%2)*n];
        }
    } dist(short_str);
    for(std::size_t y = 1; y <= long_str.size(); ++y){
        for(std::size_t x = 1; x <= short_str.size(); ++x){
            dist.set(x, y, (std::min)((std::min)(
                dist(x - 1, y)     + 1,
                dist(x,     y - 1) + 1),
                dist(x - 1, y - 1) + (short_str[x-1] == long_str[y-1] ? 0 : 1)
            ));
        }
    }
    return dist(short_str.size(), long_str.size());
}
namespace detail{
template<class XY, class LoopEnd, class DeleteCord, class InsertCord, class ReplaceCord>
struct CordSet{
    XY xy;
    LoopEnd loopend;
    DeleteCord del;
    InsertCord ins;
    ReplaceCord rep;
};

template<class XY, class LoopEnd, class DeleteCord, class InsertCord, class ReplaceCord>
CordSet(XY, LoopEnd, DeleteCord, InsertCord, ReplaceCord) -> CordSet<XY, LoopEnd, DeleteCord, InsertCord, ReplaceCord>;

constexpr CordSet x_axis{
    [](std::size_t diag, std::size_t n, std::size_t short_len){ return std::make_tuple(diag - n - 1, n - 1); },
    [](std::size_t diag, std::size_t short_len, std::size_t long_len){ return std::min(short_len, diag); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 1, n - 1); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 1, n); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 2, n - 1); }
};

constexpr CordSet flap_back{
    [](std::size_t diag, std::size_t n, std::size_t short_len){ return std::make_tuple(short_len - n - 1, n);},
    [](std::size_t diag, std::size_t short_len, std::size_t long_len){ return short_len; },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 1, n); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 1, n + 1); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 2, n); }
};

constexpr CordSet y_axis{
    [](std::size_t diag, std::size_t n, std::size_t short_len){ return std::make_tuple(short_len - n - 1, diag - short_len + n - 1);},
    [](std::size_t diag, std::size_t short_len, std::size_t long_len){ return std::min(short_len, short_len + long_len + 1 - diag); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 1,n); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 1, n + 1); },
    [](const auto& cord_to_idx, std::size_t diag, std::size_t n){ return cord_to_idx(diag - 2, n + 1); }
};

template<typename Cord, typename Container, typename CordToIdx>
// requires (sizeof(typename Container::value_type) <= sizeof(uint32_t))
void do_scalar(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const Container& short_str, const Container& long_str, const CordToIdx& cord_to_idx){
    for(std::size_t n = elm_begin; n < cord.loopend(diag, short_str.size(), long_str.size()); ++n){
        auto [x, y] = cord.xy(diag, n, short_str.size());
        dist[cord_to_idx(diag, n)] = (std::min)((std::min)(
            dist[cord.del(cord_to_idx, diag, n)] + 1,  // delele
            dist[cord.ins(cord_to_idx, diag, n)] + 1), // insert
            dist[cord.rep(cord_to_idx, diag, n)] + (short_str[short_str.size()-1 - x] == long_str[y] ? 0 : 1) // replace
        );
    }
}


/*
template<typename T, typename Container>
do_x_axis(std::integral_constant<1>, std::vector<T>& dist, std::size_t elm_begin, std::size_t diagonal, const Container& short_str, const Container& long_str, const auto& cord_to_idx){
    for(std::size_t n = elm_begin; n < std::min(short_str.size(), diagonal); ++n){
        std::size_t x = diagonal - n - 1;
        std::size_t y = n - 1;
        dist[cord_to_idx(diagonal, n)] = (std::min)((std::min)(
            dist[cord_to_idx(diagonal - 1, n - 1)] + 1,  // delele
            dist[cord_to_idx(diagonal - 1, n    )] + 1), // insert
            dist[cord_to_idx(diagonal - 2, n - 1)] + (short_str[x] == long_str[y] ? 0 : 1) // replace
        );
    }
}

template<typename T, typename Container>
do_flap_back(std::integral_constant<1>, std::vector<T>& dist, std::size_t elm_begin, std::size_t diagonal, const Container& short_str, const Container& long_str, const auto& cord_to_idx){
    for(std::size_t n = elm_begin; n < short_str.size(); ++n){
        std::size_t x = short_str.size() - n - 1;
        std::size_t y = n;
        dist[cord_to_idx(diagonal, n)] = (std::min)((std::min)(
            dist[cord_to_idx(diagonal - 1, n    )] + 1,
            dist[cord_to_idx(diagonal - 1, n + 1)] + 1),
            dist[cord_to_idx(diagonal - 2, n)] + (short_str[x] == long_str[y] ? 0 : 1)
        );
    }
}

template<typename T, typename Container>
do_y_axis(std::integral_constant<1>, std::vector<T>& dist, std::size_t elm_begin, std::size_t diagonal, const Container& short_str, const Container& long_str, const auto& cord_to_idx){
    for(std::size_t n = elm_begin; n < std::min(short_str.size(), (short_str.size() + long_str.size()+1) - diagonal); ++n){
        std::size_t x = short_str.size() - n - 1;
        std::size_t y = diagonal - short_str.size() + n - 1;
        dist[cord_to_idx(diagonal, n)] = (std::min)((std::min)(
            dist[cord_to_idx(diagonal - 1, n    )] + 1,
            dist[cord_to_idx(diagonal - 1, n + 1)] + 1),
            dist[cord_to_idx(diagonal - 2, n + 1)] + (short_str[x] == long_str[y] ? 0 : 1)
        );
    }
}
*/

#ifdef __SSE4_1__
template<typename Cord, typename Container, typename CordToIdx>
void do_sse(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const Container& short_str, const Container& long_str, const CordToIdx& cord_to_idx){
    constexpr std::size_t lane = 4; // i32x4
    std::size_t n = elm_begin;
    
    for(; n + lane-1 < cord.loopend(diag, short_str.size(), long_str.size()); n+=lane){
        auto [x, y] = cord.xy(diag, n, short_str.size());
        __m128i del = _mm_add_epi32(
            _mm_loadu_si128(reinterpret_cast<__m128i*>(&dist[cord.del(cord_to_idx, diag, n)])),
            _mm_set1_epi32(1)
        ); // delete
        __m128i ins = _mm_add_epi32(
            _mm_loadu_si128(reinterpret_cast<__m128i*>(&dist[cord.ins(cord_to_idx, diag, n)])),
            _mm_set1_epi32(1)
        ); // insert
        __m128i rep = _mm_loadu_si128(reinterpret_cast<__m128i*>(&dist[cord.rep(cord_to_idx, diag, n)])); // just load replace
        if constexpr(sizeof(typename Container::value_type) == sizeof(uint32_t)){
            rep = _mm_add_epi32(
                rep,
                _mm_andnot_si128(_mm_cmpeq_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(&short_str[short_str.size()-1 - x])), // reversed
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(&long_str[y]))
                ), _mm_set1_epi32(1))
            ); 
        }
        else if constexpr(sizeof(typename Container::value_type) == sizeof(uint16_t)){
            rep = _mm_add_epi32(
                rep,
                _mm_andnot_si128(_mm_cmpeq_epi32(
                    _mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&short_str[short_str.size()-1 - x]))), // reversed
                    _mm_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&long_str[y])))
                ), _mm_set1_epi32(1))
            );
        }
        else if constexpr(sizeof(typename Container::value_type) == sizeof(uint8_t)){
            rep = _mm_add_epi32(
                rep,
                _mm_andnot_si128(_mm_cmpeq_epi32(
                    _mm_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&short_str[short_str.size()-1 - x]))), // reversed
                    _mm_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&long_str[y])))
                ), _mm_set1_epi32(1))
            ); 
        }
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(&dist[cord_to_idx(diag, n)]),
            _mm_min_epi32(_mm_min_epi32(del, ins), rep)
        );
    }
    do_scalar(cord, dist, n, diag, short_str, long_str, cord_to_idx);
}
#endif

#ifdef __AVX2__
template<typename Cord, typename Container, typename CordToIdx>
void do_avx(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const Container& short_str, const Container& long_str, const CordToIdx& cord_to_idx){
    constexpr std::size_t lane = 8; // i32x8
    std::size_t n = elm_begin;

    //const auto reverse = [](__m256i i){ return _mm256_shuffle_epi32(_mm256_permute2x128_si256(i,i,0b00000011),0b00011011); };
    for(; n + lane-1 < cord.loopend(diag, short_str.size(), long_str.size()); n+=lane){
        auto [x, y] = cord.xy(diag, n, short_str.size());
        __m256i del = _mm256_add_epi32(
            _mm256_loadu_si256(reinterpret_cast<__m256i*>(&dist[cord.del(cord_to_idx, diag, n)])),
            _mm256_set1_epi32(1)
        ); // delete
        __m256i ins = _mm256_add_epi32(
            _mm256_loadu_si256(reinterpret_cast<__m256i*>(&dist[cord.ins(cord_to_idx, diag, n)])),
            _mm256_set1_epi32(1)
        ); // insert
        __m256i rep = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&dist[cord.rep(cord_to_idx, diag, n)])); // just load replace
        if constexpr(sizeof(typename Container::value_type) == sizeof(uint32_t)){
            rep = _mm256_add_epi32(
                rep,
                _mm256_andnot_si256(_mm256_cmpeq_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&short_str[short_str.size()-1 - x])), // reversed
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&long_str[y]))
                ), _mm256_set1_epi32(1))
            ); 
        }
        else if constexpr(sizeof(typename Container::value_type) == sizeof(uint16_t)){
            rep = _mm256_add_epi32(
                rep,
                _mm256_andnot_si256(_mm256_cmpeq_epi32(
                    _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&short_str[short_str.size()-1 - x]))), // reversed
                    _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&long_str[y])))
                ), _mm256_set1_epi32(1))
            );
        }
        else if constexpr(sizeof(typename Container::value_type) == sizeof(uint8_t)){
            rep = _mm256_add_epi32(
                rep,
                _mm256_andnot_si256(_mm256_cmpeq_epi32(
                    _mm256_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&short_str[short_str.size()-1 - x]))), // reversed
                    _mm256_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&long_str[y])))
                ), _mm256_set1_epi32(1))
            ); 
        }
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(&dist[cord_to_idx(diag, n)]),
            _mm256_min_epi32(_mm256_min_epi32(del, ins), rep)
        );
    }

    do_sse(cord, dist, n, diag, short_str, long_str, cord_to_idx);
}
#endif
} // namespace detail

template<typename Container>
uint32_t levenshtein_distance_simd(const Container& str1, const Container& str2){
    static_assert(
        sizeof(typename Container::value_type) == sizeof(int32_t)
        || sizeof(typename Container::value_type) == sizeof(int16_t)
        || sizeof(typename Container::value_type) == sizeof(int8_t),
    "unsuported value type");
    if(str1.size() + str2.size() > static_cast<typename Container::size_type>(std::numeric_limits<int32_t>::max())){
        throw std::runtime_error("Given container size is too big.");
    }
    const auto [short_str_view, long_str_view] = (str1.size() < str2.size() ? std::tie(str1,str2) : std::tie(str2, str1));
    #ifdef __SSE4_1__
    if(short_str_view.size() < 8) return levenshtein_distance_nosimd(short_str_view, long_str_view);
    #else
    return levenshtein_distance_nosimd(short_str_view, long_str_view);
    #endif
    const auto short_str = [](const auto& short_str_view) {
        std::vector<typename Container::value_type> str(short_str_view.size() + 8 - short_str_view.size() % 8);
        str.resize(short_str_view.size());
        std::reverse_copy(short_str_view.cbegin(), short_str_view.cend(), str.begin());
        return str;
    }(short_str_view);
    const auto long_str = [](const auto& long_str_view) {
        std::vector<typename Container::value_type> str(long_str_view.cbegin(), long_str_view.cend());
        str.reserve(long_str_view.size() + 8 - long_str_view.size() % 8);
        return str;
    }(long_str_view);
    
    std::vector<uint32_t> dist((short_str.size() + 1) * 3, 0);
    const auto cord_to_idx = [w = short_str.size() + 1](std::size_t diagonal, std::size_t elm){
        return (diagonal%3)*w + elm;
    };
    // initialize
    dist[cord_to_idx(1, 0)] = 1;
    dist[cord_to_idx(1, 1)] = 1;
    // x axis
    std::size_t diagonal = 2;
    for(; diagonal <= short_str.size(); ++diagonal){
        dist[cord_to_idx(diagonal, 0)] = diagonal;
        dist[cord_to_idx(diagonal, diagonal)] = diagonal;
    
        #ifdef __AVX2__
        detail::do_avx(detail::x_axis, dist, 1,  diagonal, short_str, long_str, cord_to_idx);
        #elif defined(__SSE4_1__)
        detail::do_sse(detail::x_axis, dist, 1,  diagonal, short_str, long_str, cord_to_idx);
        #endif
    }
    // flap back
    if(diagonal <= long_str.size()) dist[cord_to_idx(diagonal, short_str.size())] = diagonal;
    #ifdef __AVX2__
    detail::do_avx(detail::flap_back, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
    #elif defined(__SSE4_1__)
    detail::do_sse(detail::flap_back, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
    #endif
    ++diagonal;
    // y axis
    for(; diagonal < (short_str.size() + long_str.size() + 1); ++diagonal){
        if(diagonal <= long_str.size()) dist[cord_to_idx(diagonal, short_str.size())] = diagonal;
        #ifdef __AVX2__
        detail::do_avx(detail::y_axis, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
        #elif defined(__SSE4_1__)
        detail::do_sse(detail::y_axis, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
        #endif
    }
    
    return dist[cord_to_idx(short_str.size() + long_str.size(), 0)];
}
} // namespace LevenshteinDistansAVX2
