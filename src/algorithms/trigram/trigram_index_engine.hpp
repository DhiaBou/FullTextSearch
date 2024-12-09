#ifndef TRIGRAM_INDEX_ENGINE_HPP
#define TRIGRAM_INDEX_ENGINE_HPP
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
  std::vector<DocumentID> search(const std::string &query,
                                 const scoring::ScoringFunction &score_func) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
  /// The underlying index.
  trigramlib::HashIndex<16> index;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAM_INDEX_ENGINE_HPP
