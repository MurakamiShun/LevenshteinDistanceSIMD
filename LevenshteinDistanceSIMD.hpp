/*
 * LevenshteinDistansSIMD: https://github.com/MurakamiShun/LevenshteinDistanceSIMD
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

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace LevenshteinDistansSIMD{
/*
uint64_t levenshtein_distance_debug(const std::string_view str1, const std::string_view str2){
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

    return dist[cord_to_idx(str1.size(), str2.size())];
}
*/

template<typename Container>
uint32_t levenshtein_distance_nosimd(const Container& str1, const Container& str2){
    const auto [short_str, long_str] = (str1.size() < str2.size() ? std::tie(str1,str2) : std::tie(str2, str1));
    struct Dist{
        std::vector<uint32_t> dist_vec;
        std::array<uint32_t,16> dist_array;
        uint32_t* dist;
        const std::size_t n;
        Dist(const Container& str):
        n(str.size()){
            if(n > 8){
                dist_vec = std::vector<uint32_t>(n * 2);
                dist = dist_vec.data();
            }
            else dist = dist_array.data();
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
        uint32_t latest_set = y;
        for(std::size_t x = 1; x <= short_str.size(); ++x){
            latest_set = (std::min)((std::min)(
                latest_set + 1,
                dist(x,     y - 1) + 1),
                dist(x - 1, y - 1) + (short_str[x-1] == long_str[y-1] ? 0 : 1)
            );
            dist.set(x, y, latest_set);
        }
    }
    return dist(short_str.size(), long_str.size());
}

namespace detail{
template<typename Container>
uint32_t levenshtein_distance_very_small(const Container& short_str, const Container& long_str){
    #ifdef __SSE4_1__
    constexpr auto max_short_size = 6;
    #elif defined(__ARM_NEON)
    constexpr auto max_short_size = 8;
    #endif
    std::array<std::size_t, max_short_size*2+2> data;
    for(std::size_t x = 0; x <= short_str.size(); ++x) data[x* 2] = x;

    for(std::size_t y = 1; y <= long_str.size(); ++y){
        auto latest_set = data[y&1] = y;
        struct {
            std::size_t set,ins,rep;
        } idxs = {2+(y&1), 2+((~y)&1), (~y)&1};
        for(std::size_t x = 0; x < short_str.size(); ++x){
            latest_set = data[idxs.set] = (std::min)((std::min)(
                latest_set + 1,
                data[idxs.ins] + 1),
                data[idxs.rep] + (short_str[x] == long_str[y-1] ? 0 : 1)
            );
            idxs.set += 2;
            idxs.ins += 2;
            idxs.rep += 2;
        }
    }
    return static_cast<uint32_t>(data[short_str.size()*2 + (long_str.size()&1)]);
}

#if defined(__SSE4_1__) || defined(__ARM_NEON)
template<typename Container>
uint32_t levenshtein_distance_simd_backward_and_forward(const Container& short_str, const Container& long_str){
    /*
     *       M  F  O  K  H  E  L  
     *    0  1  2  3  4  5  6  7 
     * D  1  1  →  →  →  →  →  ↙
     * B  2  2  →  3  4  5  6  7 
     * M  3  2  3  3  4  5  6  7 
     * N  4  3  3  4  4  5  6  7 
     * E  5  4  4  4  5  5  5 ← 
     * K  6  ↗  ←  ←  ←  ←  ←  6
     * calculate backward and forward
     */

    #ifdef __SSE4_1__
    constexpr auto reserve_short_size = 12;
    constexpr auto reserve_long_size = 16;
    struct alignas(16) x2int : std::array<uint64_t, 2>{};
    #elif defined(__ARM_NEON)
    constexpr auto reserve_short_size = 12;
    constexpr auto reserve_long_size = 24;
    struct alignas(16) x2int : std::array<uint32_t, 2>{};
    #endif
    std::array<x2int, reserve_short_size*2+2> data_array;
    std::vector<x2int> data_vec;
    x2int* const data = [&](){
        if(reserve_short_size < short_str.size()){
            data_vec.resize(short_str.size()*2+2);
            return data_vec.data();
        }
        else{
            return data_array.data();
        }
    }();
    std::array<x2int, reserve_short_size> short_str_copy_array;
    std::array<x2int, reserve_long_size> long_str_copy_array;
    std::vector<x2int> short_str_copy_vec;
    std::vector<x2int> long_str_copy_vec;
    x2int* const short_str_copy = [&](){
        if(reserve_short_size < short_str.size()){
            short_str_copy_vec.resize(short_str.size());
            return short_str_copy_vec.data();
        }
        else{
            return short_str_copy_array.data();
        }
    }();
    x2int* const long_str_copy = [&](){
        if(reserve_long_size < long_str.size()){
            long_str_copy_vec.resize(long_str.size());
            return long_str_copy_vec.data();
        }
        else{
            return long_str_copy_array.data();
        }
    }();
    const auto is_odd = long_str.size()&1;
    
    for(std::size_t i = 0, r = short_str.size() - 1; i < short_str.size()/2 + (short_str.size()&1); ++i, --r){
        short_str_copy[i] = x2int{
            static_cast<typename x2int::value_type>(short_str[r]),
            static_cast<typename x2int::value_type>(short_str[i])
        };
        short_str_copy[r] = x2int{ short_str_copy[i][1], short_str_copy[i][0] };
    }
    
    for(std::size_t i = 0, r = long_str.size() - 1; i < long_str.size()/2 + is_odd; ++i, --r){
        long_str_copy[i] = x2int{
            static_cast<typename x2int::value_type>(long_str[r]),
            static_cast<typename x2int::value_type>(long_str[i])
        };
        long_str_copy[r] = x2int{ long_str_copy[i][1], long_str_copy[i][0] };
    }
    
    #ifdef __SSE4_1__
    for(std::size_t x = 0; x <= short_str.size(); ++x){
        _mm_store_si128(reinterpret_cast<__m128i*>(&data[x*2]), _mm_set1_epi64x(x));
    }

    for(std::size_t y = 1; y <= long_str.size()/2 + is_odd; ++y){
        const auto long_chars = _mm_load_si128(reinterpret_cast<__m128i*>(&long_str_copy[y-1]));
        struct {
            std::size_t set,ins,rep;
        } idxs = {2+(y&1), 2+((~y)&1), (~y)&1};

        auto latest_set = _mm_set1_epi64x(y);
        _mm_store_si128(reinterpret_cast<__m128i*>(&data[y&1]), latest_set);
        for(std::size_t x = 0; x < short_str.size(); ++x){
            const auto short_chars = _mm_load_si128(reinterpret_cast<__m128i*>(&short_str_copy[x]));
            const auto del = _mm_add_epi64(latest_set, _mm_set1_epi64x(1));
            const auto ins = _mm_add_epi64(_mm_load_si128(reinterpret_cast<__m128i*>(&data[idxs.ins])), _mm_set1_epi64x(1));
            const auto rep = _mm_add_epi64(
                _mm_load_si128(reinterpret_cast<__m128i*>(&data[idxs.rep])),
                _mm_andnot_si128(
                    _mm_cmpeq_epi64(short_chars, long_chars),
                    _mm_set1_epi64x(1)
                )
            );
            // distance < uint32_max
            latest_set = _mm_min_epu32(_mm_min_epu32(
                del,
                ins),
                rep
            );
            _mm_store_si128(reinterpret_cast<__m128i*>(&data[idxs.set]), latest_set);
            idxs.set += 2;
            idxs.ins += 2;
            idxs.rep += 2;
        }
    }
    #elif defined(__ARM_NEON)
    for(std::size_t x = 0; x <= short_str.size(); ++x){
        vst1_u32(&data[x*2][0], vmov_n_u32(x));
    }

    for(std::size_t y = 1; y <= long_str.size()/2 + is_odd; ++y){
        const auto long_chars = vld1_u32(&long_str_copy[y-1][0]);
        struct {
            std::size_t set,ins,rep;
        } idxs = {2+(y&1), 2+((~y)&1), (~y)&1};
        auto latest_set = vmov_n_u32(y);
        vst1_u32(&data[y&1][0], latest_set);
        for(std::size_t x = 0; x < short_str.size(); ++x){
            const auto short_chars = vld1_u32(&short_str_copy[x][0]);
            const auto del = vadd_u32(latest_set, vmov_n_u32(1));
            const auto ins = vadd_u32(vld1_u32(&data[idxs.ins][0]), vmov_n_u32(1));
            const auto rep = vadd_u32(
                vld1_u32(&data[idxs.rep][0]),
                vand_u32(
                    vmvn_u32(vceq_u32(short_chars, long_chars)),
                    vmov_n_u32(1)
                )
            );
            // distance < uint32_max
            latest_set = vmin_u32(vmin_u32(
                del,
                ins),
                rep
            );
            vst1_u32(&data[idxs.set][0], latest_set);
            idxs.set += 2;
            idxs.ins += 2;
            idxs.rep += 2;
        }
    }
    #endif
    
    const auto y = long_str.size()/2;
    auto min_value = data[(y+is_odd)&1][1] + data[short_str.size()*2 + (y&1)][0];
    for(std::size_t x = 1; x <= short_str.size(); ++x){
        const auto val = data[x*2 + ((y+is_odd)&1)][1] + data[(short_str.size() - x)*2 + (y&1)][0];
        min_value = (std::min)(min_value, val);
    }
    return min_value;
}
#endif

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
void do_scalar(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const Container& short_str, const Container& long_str, const CordToIdx& cord_to_idx){
    const auto end_idx = cord.loopend(diag, short_str.size(), long_str.size());
    for(std::size_t n = elm_begin; n < end_idx; ++n){
        auto [x, y] = cord.xy(diag, n, short_str.size());
        dist[cord_to_idx(diag, n)] = (std::min)((std::min)(
            dist[cord.del(cord_to_idx, diag, n)] + 1,  // delele
            dist[cord.ins(cord_to_idx, diag, n)] + 1), // insert
            dist[cord.rep(cord_to_idx, diag, n)] + (short_str[short_str.size()-1 - x] == long_str[y] ? 0 : 1) // replace
        );
    }
}
#ifdef __ARM_NEON
template<typename Cord, typename CordToIdx>
void do_neon(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const std::vector<uint32_t>& short_str, const std::vector<uint32_t>& long_str, const CordToIdx& cord_to_idx){
    constexpr std::size_t lane = 4; // i32x4
    const auto end_idx = cord.loopend(diag, short_str.size(), long_str.size());

    for(std::size_t n = elm_begin; n + lane <= end_idx; n+=lane){
        auto [x, y] = cord.xy(diag, n, short_str.size());        
        uint32x4_t del = vaddq_u32(
            vld1q_u32(&dist[cord.del(cord_to_idx, diag, n)]),
            vmovq_n_u32(1)
        ); // delete
        uint32x4_t ins = vaddq_u32(
            vld1q_u32(&dist[cord.ins(cord_to_idx, diag, n)]),
            vmovq_n_u32(1)
        ); // insert
        uint32x4_t rep = vld1q_u32(&dist[cord.rep(cord_to_idx, diag, n)]); // just load replace
        rep = vaddq_u32(
            rep,
            vandq_u32(vmvnq_u32(vceqq_u32(
                vld1q_u32(reinterpret_cast<const uint32_t*>(&short_str[short_str.size()-1 - x])), // reversed
                vld1q_u32(reinterpret_cast<const uint32_t*>(&long_str[y]))
            )), vmovq_n_u32(1))
        ); 
        
        vst1q_u32(
            &dist[cord_to_idx(diag, n)],
            vminq_u32(vminq_u32(del, ins), rep)
        );
    }
    const auto next_idx = end_idx - (end_idx - elm_begin) % lane;
    do_scalar(cord, dist, next_idx, diag, short_str, long_str, cord_to_idx);
}
#endif
#ifdef __SSE4_1__
template<typename Cord, typename CordToIdx>
void do_sse(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const std::vector<uint32_t>& short_str, const std::vector<uint32_t>& long_str, const CordToIdx& cord_to_idx){
    constexpr std::size_t lane = 4; // i32x4
    const auto end_idx = cord.loopend(diag, short_str.size(), long_str.size());

    for(std::size_t n = elm_begin; n + lane <= end_idx; n+=lane){
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
        rep = _mm_add_epi32(
            rep,
            _mm_andnot_si128(_mm_cmpeq_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&short_str[short_str.size()-1 - x])), // reversed
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(&long_str[y]))
            ), _mm_set1_epi32(1))
        );
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(&dist[cord_to_idx(diag, n)]),
            _mm_min_epi32(_mm_min_epi32(del, ins), rep)
        );
    }
    const auto next_idx = end_idx - (end_idx - elm_begin) % lane;
    do_scalar(cord, dist, next_idx, diag, short_str, long_str, cord_to_idx);
}
#endif
#ifdef __AVX2__
template<typename Cord, typename CordToIdx>
void do_avx(const Cord cord, std::vector<uint32_t>& dist, std::size_t elm_begin, std::size_t diag, const std::vector<uint32_t>& short_str, const std::vector<uint32_t>& long_str, const CordToIdx& cord_to_idx){
    constexpr std::size_t lane = 8; // i32x8

    const auto end_idx = cord.loopend(diag, short_str.size(), long_str.size());
    for(std::size_t n = elm_begin; n + lane <= end_idx; n+=lane){
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
        rep = _mm256_add_epi32(
            rep,
            _mm256_andnot_si256(_mm256_cmpeq_epi32(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&short_str[short_str.size()-1 - x])), // reversed
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&long_str[y]))
            ), _mm256_set1_epi32(1))
        ); 
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(&dist[cord_to_idx(diag, n)]),
            _mm256_min_epi32(_mm256_min_epi32(del, ins), rep)
        );
    }
    const auto next_idx = end_idx - (end_idx - elm_begin) % lane;
    do_sse(cord, dist, next_idx, diag, short_str, long_str, cord_to_idx);
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
    if(short_str_view.size() < 48){
        if(long_str_view.size() < 8) return detail::levenshtein_distance_very_small(short_str_view, long_str_view);
        return detail::levenshtein_distance_simd_backward_and_forward(short_str_view, long_str_view);
    }
    #elif defined(__ARM_NEON)
    if(short_str_view.size() < 32){
        if(long_str_view.size() < 8) return detail::levenshtein_distance_very_small(short_str_view, long_str_view);
        return detail::levenshtein_distance_simd_backward_and_forward(short_str_view, long_str_view);
    }
    #else
    return levenshtein_distance_nosimd(short_str_view, long_str_view);
    #endif
    
    const auto short_str = std::vector<uint32_t>(std::reverse_iterator(short_str_view.cend()), std::reverse_iterator(short_str_view.cbegin()));
    const auto long_str = std::vector<uint32_t>(long_str_view.cbegin(), long_str_view.cend());
    
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
        #elif defined(__ARM_NEON)
        detail::do_neon(detail::x_axis, dist, 1,  diagonal, short_str, long_str, cord_to_idx);
        #else
        detail::do_scalar(detail::x_axis, dist, 1,  diagonal, short_str, long_str, cord_to_idx);
        #endif
        
    }
    // flap back
    
    if(diagonal <= long_str.size()) dist[cord_to_idx(diagonal, short_str.size())] = diagonal;
    #ifdef __AVX2__
    detail::do_avx(detail::flap_back, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
    #elif defined(__SSE4_1__)
    detail::do_sse(detail::flap_back, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
    #elif defined(__ARM_NEON)
    detail::do_neon(detail::flap_back, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
    #else
    detail::do_scalar(detail::flap_back, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
    #endif
    
    ++diagonal;
    // y axis
    for(; diagonal < (short_str.size() + long_str.size() + 1); ++diagonal){
        if(diagonal <= long_str.size()) dist[cord_to_idx(diagonal, short_str.size())] = diagonal;
        
        #ifdef __AVX2__
        detail::do_avx(detail::y_axis, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
        #elif defined(__SSE4_1__)
        detail::do_sse(detail::y_axis, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
        #elif defined(__ARM_NEON)
        detail::do_neon(detail::y_axis, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
        #else
        detail::do_scalar(detail::y_axis, dist, 0,  diagonal, short_str, long_str, cord_to_idx);
        #endif
    }
    
    return dist[cord_to_idx(short_str.size() + long_str.size(), 0)];
}
} // namespace LevenshteinDistansSIMD
