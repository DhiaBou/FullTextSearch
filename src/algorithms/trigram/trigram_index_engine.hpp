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
  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
  /// The underlying index.
  trigramlib::HashIndex<16> index;
  /// The number of indexed documents.
  uint32_t doc_count;
  /// A map from document ID to document length in trigrams.
  std::unordered_map<DocumentID, uint32_t> doc_to_length;
  /// The average document length in trigrams.
  double avg_doc_length;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAM_INDEX_ENGINE_HPP
