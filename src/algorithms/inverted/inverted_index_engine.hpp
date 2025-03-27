//
// Created by dhia on 12/6/24.
//

#ifndef INVERTED_INDEX_ENGINE_HPP
#define INVERTED_INDEX_ENGINE_HPP

#include <thread>

#include "data-structures/parallel_hash_table.hpp"
#include "documents/document_iterator.hpp"
#include "fts_engine.hpp"

struct Token;

class InvertedIndexEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(std::string &data_path) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;
  uint64_t footprint() override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
  void estimateDataStructureSizes(const std::string &data_path);

  void indexBatch(const std::vector<Document> &batch);

  const uint64_t NUM_THREADS = std::thread::hardware_concurrency();

  double average_doc_length_ = -1.0;

  /// key is token, value is a map of doc id to term frequency
  ParallelHashTable<std::string, std::vector<std::pair<DocumentID, uint32_t>>>
      term_frequency_per_document_{1};

  /// key is document id, value is number of tokens or terms
  std::vector<uint32_t> tokens_per_document_;
};

#endif  // INVERTED_INDEX_ENGINE_HPP
