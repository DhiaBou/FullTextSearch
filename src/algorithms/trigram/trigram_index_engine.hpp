#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
#include <unordered_set>
//---------------------------------------------------------------------------
#include "../../fts_engine.hpp"
#include "index/hash_index.hpp"
#include "models/trigram.hpp"
//---------------------------------------------------------------------------
class TrigramIndexEngine : public FullTextSearchEngine {
 public:
  /// Build the index.
  void indexDocuments(DocumentIterator it) override;
  /// Search for string.
  std::vector<uint32_t> search(const std::string &query) override;

 private:
  /// The underlying index.
  HashIndex<16> index;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAMINDEXENGINE_HPP
