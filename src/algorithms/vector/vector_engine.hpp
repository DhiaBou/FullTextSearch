//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include <sys/types.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../fts_engine.hpp"

class VectorEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(DocumentIterator it) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
  // prints how often a token occurs in this vector
  void print_vector(std::vector<double> v);
  std::vector<double> compress_vector(std::vector<double> v);
  std::vector<double> decompress_vector(std::vector<double> v);
  void store_vectors();
  void load_vectors();

  /// key is token, value is a map of doc id to term frequency
  std::unordered_map<std::string, std::unordered_map<DocumentID, uint32_t>>
      term_frequency_per_document_;

  /// key is document id, value is number of tokens or terms
  std::unordered_map<DocumentID, uint32_t> tokens_per_document_;

  /// key is token, value is number of documents this token appears in
  std::unordered_map<std::string, uint32_t> documents_per_token_;

  /// key is document id, value is the vector of the document
  std::unordered_map<DocumentID, std::vector<double>> document_to_vector_;
};

#endif  // VECTORSPACEMODELENGINE_HPP
