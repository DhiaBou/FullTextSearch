#ifndef TRIGRAM_INDEX_ENGINE_HPP
#define TRIGRAM_INDEX_ENGINE_HPP
//---------------------------------------------------------------------------
#include "../../documents/document_iterator.hpp"
#include "../../fts_engine.hpp"
#include "index/hash_index.hpp"
#include "index/parallel_hash_index.hpp"
#include "models/trigram.hpp"
#include "parser/trigram_parser.hpp"
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
  /// Determines the memory footprint of the engine in bytes.
  uint64_t footprint() override;

  /// Get the number of documents.
  uint32_t getDocumentCount() override;
  /// Get the average length of a document.
  double getAvgDocumentLength() override;

 private:
  /// @brief Merges given doc_to_length maps into one.
  /// To be precise, merges the given maps into the doc_to_length vector.
  /// @param maps The maps to be merged.
  void merge(std::vector<std::unordered_map<DocumentID, uint32_t>> &maps);
  /// @brief Consumes documents and writes trigrams into the index.
  /// @param doc_it The iterator to consume the documents from.
  /// @param local_doc_to_length The index to store the number of trigrams per document.
  /// @return The total number of found trigrams.
  uint64_t consumeDocuments(DocumentIterator &doc_it,
                            std::unordered_map<DocumentID, uint32_t> &local_doc_to_length);

  /// The underlying index.
  trigramlib::ParallelHashIndex<trigramlib::kNumPossibleTrigrams, trigramlib::kMaxWordOffset> index;
  /// The number of indexed documents.
  std::atomic<uint32_t> doc_count;
  /// A mapping from document ID (index) to document length in trigrams.
  /// Note: We make use of the fact that document IDs are increasing consecutively,
  /// starting at 1.
  std::vector<uint32_t> doc_to_length;
  /// The average document length in trigrams.
  double avg_doc_length;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAM_INDEX_ENGINE_HPP
