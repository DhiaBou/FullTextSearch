#ifndef HYPERLOGLOG_H
#define HYPERLOGLOG_H

#include <cstdint>
#include <vector>

template <typename T>
class HyperLogLog {
  static constexpr uint64_t CONSIDERED_TRAILING_BITS = 7;
  static constexpr uint64_t NUM_REGISTERS = 1ULL << CONSIDERED_TRAILING_BITS;
  static constexpr double ALPHA_M = 0.7213 / (1.0 + 1.079 / NUM_REGISTERS);

  double harmonicMean(std::vector<uint64_t> &vals) {
    double denominator = 0.0;

    for (const uint64_t val : vals) {
      denominator += (1.0 / static_cast<double>(1ULL << val));
    }

    return static_cast<double>(vals.size()) / denominator;
  }

 public:
  explicit HyperLogLog(size_t num_threads) {
    trailing_zeros = std::vector(num_threads, std::vector<uint64_t>(NUM_REGISTERS, 0));
  }

  void add(T val, size_t thread_id) {
    const uint64_t hash_val = std::hash<T>()(val);

    const uint64_t slot = hash_val >> (64 - CONSIDERED_TRAILING_BITS);
    const uint64_t num_zeros = hash_val == 0 ? 64 : static_cast<uint64_t>(__builtin_ctz(hash_val));

    trailing_zeros[thread_id][slot] = std::max(trailing_zeros[thread_id][slot], num_zeros);
  }

  uint64_t getCount() {
    std::vector<uint64_t> res(NUM_REGISTERS, 0);

    for (size_t i = 0; i < trailing_zeros.size(); ++i) {
      for (size_t j = 0; j < trailing_zeros[i].size(); ++j) {
        res[j] = std::max(res[j], trailing_zeros[i][j] + 1);
      }
    }

    return static_cast<uint64_t>(ALPHA_M * NUM_REGISTERS * harmonicMean(res));
  }

 private:
  std::vector<std::vector<uint64_t>> trailing_zeros;
};

#endif  // HYPERLOGLOG_H
