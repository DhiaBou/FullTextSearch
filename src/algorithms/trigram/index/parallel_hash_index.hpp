#ifndef TRIGRAM_PARALLEL_HASH_INDEX_HPP
#define TRIGRAM_PARALLEL_HASH_INDEX_HPP
//---------------------------------------------------------------------------
#include "data-structures/parallel_hash_table.hpp"
#include "index.hpp"
#include "models/doc_freq.hpp"
#include "models/trigram.hpp"
//---------------------------------------------------------------------------
namespace trigramlib {
template <size_t TableSize, uint8_t MaxOffset>
class ParallelHashIndex : public Index<DocFreq, std::vector<DocFreq>, MaxOffset> {
 public:
  /// Constructor.
  ParallelHashIndex() = default;
  /// Move Constructor.
  ParallelHashIndex(ParallelHashIndex&& other) noexcept : table(std::move(other.table)) {}
  /// Move assignment.
  ParallelHashIndex& operator=(ParallelHashIndex&& other) noexcept {
    if (this != &other) {
      table = std::move(other.table);
    }
    return *this;
  }
  /// Destructor.
  ~ParallelHashIndex() override = default;
  //---------------------------------------------------------------------------
  void insert(Trigram key, DocFreq value) override {
    uint8_t offset = key.getWordOffset();
    if (offset >= MaxOffset) {
      key.setWordOffset(MaxOffset - 1);
    }

    auto push_occurence = [&value](std::vector<DocFreq>& occurences) {
      occurences.push_back(value);
    };

    table.updateOrInsert(key.getRawValue(), push_occurence, std::vector<DocFreq>{});
  }
  //---------------------------------------------------------------------------
  std::vector<DocFreq>* lookup(Trigram key) override { return table.get(key.getRawValue()); }
  //---------------------------------------------------------------------------
  void store(std::ofstream& file) override {
    // TODO
  }
  //---------------------------------------------------------------------------
  void load(const char* it, const char* end) override {
    // TODO
  }
  //---------------------------------------------------------------------------
  void compactify(uint32_t max_occurences) {
    for (auto& [key, value] : table) {
      if (value.size() > max_occurences) {
        value.clear();
        value.shrink_to_fit();
      }
    }
  }

 private:
  /// A mapping of trigram to buckets.
  ParallelHashTable<uint32_t, std::vector<DocFreq>, TableSize> table;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_PARALLEL_HASH_INDEX_HPP