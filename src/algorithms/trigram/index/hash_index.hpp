#ifndef TRIGRAM_HASH_INDEX_HPP
#define TRIGRAM_HASH_INDEX_HPP
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
      uint8_t offset = key.getWordOffset();
      if (offset >= BucketSize) {
        offset = BucketSize - 1;
      }
      return &it->second.containers[offset];
    }
    return nullptr;
  }
  /**
   * Write the underlying data structure to specified file.
   *
   * Serialization format:
   *
   * Header:
   * -------
   * 1. uint32_t HashTable Size                (4 bytes)
   *
   * Entries:
   * --------
   * 2. For each entry in the hash table:
   *  a. Trigram key                          (4 bytes)
   *  b. For each container in the bucket:
   *    i.  uint32_t container_size           (4 bytes)
   *    ii. For each DocFreq in the container:
   *      - uint32_t doc_id                   (4 bytes)
   *      - uint32_t freq                     (4 bytes)
   */
  void store(std::ofstream& file) override {
    // TODO: How to deal with different BucketSizes?

    auto table_size = static_cast<uint32_t>(table.size());
    file.write(reinterpret_cast<const char*>(&table_size), sizeof(table_size));

    for (const auto& [trigram, bucket] : table) {
      file.write(reinterpret_cast<const char*>(&trigram), sizeof(trigram));
      for (const auto& container : bucket.containers) {
        auto container_size = static_cast<uint32_t>(container.size());
        file.write(reinterpret_cast<const char*>(&container_size), sizeof(container_size));
        for (const auto& doc_freq : container) {
          file.write(reinterpret_cast<const char*>(&doc_freq.doc_id), sizeof(doc_freq.doc_id));
          file.write(reinterpret_cast<const char*>(&doc_freq.freq), sizeof(doc_freq.freq));
        }
      }
    }
  }

  /// Load the underlying data structure from given data.
  void load(const char* it, const char* end) override {
    auto table_size = *reinterpret_cast<const uint32_t*>(it);
    it += sizeof(table_size);

    table.reserve(table_size);

    while (it < end) {
      auto raw_trigram = *reinterpret_cast<const uint32_t*>(it);
      Trigram trigram{raw_trigram};
      it += sizeof(raw_trigram);

      auto& bucket = table[trigram];

      for (uint8_t i = 0; i < BucketSize; ++i) {
        auto container_size = *reinterpret_cast<const uint32_t*>(it);
        it += sizeof(container_size);

        bucket.containers[i].reserve(container_size);

        for (uint32_t j = 0; j < container_size; ++j) {
          auto doc_id = *reinterpret_cast<const uint32_t*>(it);
          it += sizeof(doc_id);
          auto freq = *reinterpret_cast<const uint32_t*>(it);
          it += sizeof(freq);

          bucket.containers[i].push_back({doc_id, freq});
        }
      }
    }
  }

 private:
  /// A mapping of trigram to buckets.
  std::unordered_map<Trigram, Bucket<std::vector<DocFreq>, BucketSize>> table;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_HASH_INDEX_HPP