#include <iostream>
#include <string_view>
#include <chrono>
#include <random>
#include "LevenshteinDistanceSIMD.hpp"

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
    std::mt19937 mt{std::random_device{}()};

    const uint32_t len = 12;

    std::vector<std::tuple<std::string, std::string>> test_data;
    for(auto i=0; i < 1000000; ++i) test_data.push_back({gen(mt()%len), gen(mt()%len)});
    
    auto start = std::chrono::system_clock::now();
    int64_t total = 0;
    
    for(const auto& [str1, str2] : test_data){
        total += LevenshteinDistansSIMD::levenshtein_distance_nosimd(str1, str2);
    }
    std::cout << "time:" << (std::chrono::system_clock::now() - start).count() << std::endl;
    std::cout << total << std::endl;
    

    start = std::chrono::system_clock::now();
    total = 0;
    for(const auto& [str1, str2] : test_data){
        total += LevenshteinDistansSIMD::levenshtein_distance_simd(str1, str2);
    }
    std::cout << "time:" << (std::chrono::system_clock::now() - start).count() << std::endl;
    std::cout << total << std::endl;
}
