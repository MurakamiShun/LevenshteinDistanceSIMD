#include <iostream>
#include <string_view>
#include <chrono>
#include <random>
#include "LevenshteinDistance_AVX2.hpp"

char gen_char(){
    static std::mt19937 mt{std::random_device{}()};
    return mt()%(126-65) + 65;
}

std::string gen(std::size_t length){
    std::string buff(length, ' ');
    std::generate_n(buff.begin(), length, gen_char);
    return buff;
}

char16_t gen_char16(){
    static std::mt19937 mt{std::random_device{}()};
    return mt()%(126-65) + 65;
}

std::u16string gen16(std::size_t length){
    std::u16string buff(length, u' ');
    std::generate_n(buff.begin(), length, gen_char);
    return buff;
}

char32_t gen_char32(){
    static std::mt19937 mt{std::random_device{}()};
    return mt()%(126-65) + 65;
}

std::u32string gen32(std::size_t length){
    std::u32string buff(length, U' ');
    std::generate_n(buff.begin(), length, gen_char);
    return buff;
}
int main() {
    std::string str2;
    std::string str1;
    std::mt19937 mt{std::random_device{}()};

    const uint32_t len = 1000;

    auto start = std::chrono::system_clock::now();
    int64_t total = 0;
    for(auto i=0; i < 10000; ++i){
        str1 = gen(mt()%len+1);
        str2 = gen(mt()%len+1);
        total += LevenshteinDistansAVX2::levenshtein_distance_nosimd(str1, str2);
    }
    std::cout << (std::chrono::system_clock::now() - start).count() << std::endl;
    std::cout << total << std::endl;

    start = std::chrono::system_clock::now();
    total = 0;
    for(auto i=0; i < 10000; ++i){
        str1 = gen(mt()%len+1);
        str2 = gen(mt()%len+1);
        total += LevenshteinDistansAVX2::levenshtein_distance_simd(str1, str2);
    }
    std::cout << (std::chrono::system_clock::now() - start).count() << std::endl;
    std::cout << total << std::endl;
}
