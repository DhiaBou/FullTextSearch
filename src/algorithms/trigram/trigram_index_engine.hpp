#ifndef TRIGRAM_INDEX_ENGINE_HPP
#define TRIGRAM_INDEX_ENGINE_HPP
//---------------------------------------------------------------------------
#include "../../documents/document_iterator.hpp"
#include "../../fts_engine.hpp"
#include "index/hash_index.hpp"
#include "models/trigram.hpp"
//---------------------------------------------------------------------------
class TrigramIndexEngine : public FullTextSearchEngine {
 public:
  /// Build the index.
  void indexDocuments(std::string &data_path) override;
  /// Search for string.
  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;
  /// Store the index at specified location.
  void store(const std::string &path);
  /// Load the index from specified location.
  void load(const std::string &path);

  /// Get the number of documents.
  uint32_t getDocumentCount() override;
  /// Get the average length of a document.
  double getAvgDocumentLength() override;

 private:
  /// @brief Merges given indexes into one.
  /// To be precise, merges the given indexes into the one located at index 0.
  /// @param indexes The indexes to be merged.
  void merge(std::vector<trigramlib::HashIndex<16>> &indexes);
  /// @brief Consumes documents and writes trigrams into the index.
  /// @param doc_it The iterator to consume the documents from.
  /// @param local_index The index to write the trigrams into.
  /// @param local_doc_to_length The index to store the number of trigrams per document.
  /// @return The total number of found trigrams.
  uint64_t consumeDocuments(DocumentIterator &doc_it, trigramlib::HashIndex<16> &local_index,
                            std::unordered_map<DocumentID, uint32_t> &local_doc_to_length);

  /// The underlying index.
  trigramlib::HashIndex<16> index;
  /// The number of indexed documents.
  std::atomic<uint32_t> doc_count;
  /// A map from document ID to document length in trigrams.
  std::unordered_map<DocumentID, uint32_t> doc_to_length;
  /// The average document length in trigrams.
  double avg_doc_length;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAM_INDEX_ENGINE_HPP
