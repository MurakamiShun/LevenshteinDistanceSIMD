# LevenshteinDistanceSIMD
LevenshteinDistanceSIMD is C++17 header for calculate the Levenshtein distance faaaaast with AVX2/SSE4.1 and ASIMD(NEON).

# Usage
Call `LevenshteinDistansSIMD::levenshtein_distance_simd`.
```
template<typename Container>
uint32_t LevenshteinDistansSIMD::levenshtein_distance_simd(const Container& str1, const Container& str2){
  /* Conditions */
  static_assert(
      sizeof(typename Container::value_type) == sizeof(int32_t)
      || sizeof(typename Container::value_type) == sizeof(int16_t)
      || sizeof(typename Container::value_type) == sizeof(int8_t),
  "unsuported value type");
  if(str1.size() + str2.size() > (std::numeric_limits<int32_t>::max()){
    throw std::runtime_error("Given container size is too big.");
  }
  
  /* Calc..... */
}
```

# License
This software is released under the MIT License, see [LICENSE](LICENSE).
