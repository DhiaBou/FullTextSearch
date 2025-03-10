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
  //---------------------------------------------------------------------------
  struct Entry {
    /// Default constructor.
    Entry() : trigram(0), doc_freqs({}) {}
    /// Constructor.
    Entry(Trigram trigram, std::vector<DocFreq> doc_freqs)
        : trigram(trigram), doc_freqs(std::move(doc_freqs)) {}
    /// The entry's trigram.
    Trigram trigram;
    /// The entry's pair of document ID and frequency.
    std::vector<DocFreq> doc_freqs;
  };
  //---------------------------------------------------------------------------
  class EntryIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using MapIterator =
        std::unordered_map<Trigram, Bucket<std::vector<DocFreq>, BucketSize>>::iterator;
    using ContainerIterator = std::vector<DocFreq>::iterator;

    /// Constructor.
    EntryIterator(MapIterator begin, MapIterator end) : begin(begin), end(end) {
      if (begin != end) {
        advance();
      }
    }
    /// Dereference operator.
    Entry& operator*() { return current; }
    /// Member access operator.
    Entry* operator->() { return &current; }
    /// Pre-increment operator.
    EntryIterator& operator++() {
      advance();
      return *this;
    }
    /// Post-increment operator.
    EntryIterator operator++(int) {
      EntryIterator tmp = *this;
      ++(*this);
      return tmp;
    }
    /// Equality operator.
    bool operator==(const EntryIterator other) {
      return begin == other.begin && current_cont == other.current_cont;
    }
    /// Inequality operator.
    bool operator!=(const EntryIterator other) { return !(*this == other); }

   private:
    void advance() {
      while (begin != end) {
        if (current_cont >= BucketSize) {
          ++begin;
          current_cont = 0;
          continue;
        }
        auto& container = begin->second.containers[current_cont];
        if (container.begin() == container.end()) {
          ++current_cont;
        } else {
          Trigram trigram = begin->first;
          trigram.setWordOffset(current_cont);
          current = Entry(trigram, begin->second.containers[current_cont]);
          ++current_cont;
          return;
        }
      }
      current = Entry(Trigram(), {});
    }
    Entry current;
    MapIterator begin;
    MapIterator end;
    uint8_t current_cont = 0;
  };
  //---------------------------------------------------------------------------
  /// Constructor.
  HashIndex() = default;
  /// Move Constructor.
  HashIndex(HashIndex&& other) noexcept : table(std::move(other.table)) {}
  /// Move assignment.
  HashIndex& operator=(HashIndex&& other) noexcept {
    if (this != &other) {
      table = std::move(other.table);
    }
    return *this;
  }
  /// Destructor.
  ~HashIndex() override = default;
  /// Begin-Iterator.
  EntryIterator begin() { return EntryIterator(table.begin(), table.end()); }
  /// End-Iterator.
  EntryIterator end() { return EntryIterator(table.end(), table.end()); }
  //---------------------------------------------------------------------------
  void insert(Trigram key, DocFreq value) override {
    if (this->stop_trigrams.contains(key.getRawValue())) return;

    // Bucket Lookup
    auto& bucket = table[key];
    uint8_t offset = key.getWordOffset();
    if (offset >= BucketSize) {
      offset = BucketSize - 1;
    }
    // Container Lookup
    auto& container = bucket.containers[offset];
    if (this->stop_count > 0 && container.size() > this->stop_count) {
      this->stop_trigrams.insert(key.getRawValue());
      container.clear();
    } else {
      container.push_back(value);
    }
  }
  //---------------------------------------------------------------------------
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
  //---------------------------------------------------------------------------
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
  //---------------------------------------------------------------------------
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
  //---------------------------------------------------------------------------
  void setStopCount(uint32_t stop_count) override { this->stop_count = stop_count; }

 private:
  /// A mapping of trigram to buckets.
  std::unordered_map<Trigram, Bucket<std::vector<DocFreq>, BucketSize>> table;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_HASH_INDEX_HPP