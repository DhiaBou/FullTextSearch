#ifndef TRIGRAM_HASH_INDEX_HPP
#define TRIGRAM_HASH_INDEX_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <unordered_map>
//---------------------------------------------------------------------------
#include "index.hpp"
#include "models/doc_freq.hpp"
#include "models/trigram.hpp"
//---------------------------------------------------------------------------
namespace trigramlib {
//---------------------------------------------------------------------------
template <uint8_t BucketSize>
class HashIndex : public Index<DocFreq, std::vector<DocFreq>, BucketSize> {
 public:
  /// Constructor.
  HashIndex() = default;
  /// Destructor.
  ~HashIndex() override = default;
  /// Insert a key-value pair into the index.
  void insert(Trigram key, DocFreq value) override {
    auto& bucket = table[key];
    uint8_t offset = key.getWordOffset();
    if (offset >= BucketSize) {
      offset = BucketSize - 1;
    }
    bucket.containers[offset].push_back(value);
  }
  /// Lookup the value for given trigram.
  std::vector<DocFreq>* lookup(Trigram key) override {
    auto it = table.find(key);
    if (it != table.end()) {
      return &it->second.containers[key.getWordOffset()];
    }
    return nullptr;
  }

 private:
  /// A mapping of trigram to buckets.
  std::unordered_map<Trigram, Bucket<std::vector<DocFreq>, BucketSize>> table;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_HASH_INDEX_HPP