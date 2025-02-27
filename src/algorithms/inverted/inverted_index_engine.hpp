//
// Created by dhia on 12/6/24.
//

#ifndef INVERTED_INDEX_ENGINE_HPP
#define INVERTED_INDEX_ENGINE_HPP

#include "../../fts_engine.hpp"

struct Token;

class InvertedIndexEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(std::string &data_path) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
  double docScoreForToken(uint32_t docId, const std::string &token);

  double average_doc_length_ = -1.0;

  /// key is token, value is a map of doc id to term frequency
  std::unordered_map<std::string, std::unordered_map<DocumentID, uint32_t>>
      term_frequency_per_document_;

  /// key is document id, value is number of tokens or terms
  std::unordered_map<DocumentID, uint32_t> tokens_per_document_;
};

#endif  // INVERTED_INDEX_ENGINE_HPP
