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

using TermID = uint32_t;

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
  void print_vector(std::vector<float> values, std::vector<TermID> terms);
  std::vector<float> compress_vector(std::vector<float> v);
  std::vector<float> decompress_vector(std::vector<float> v);
  void store_vectors();
  void load_vectors();

  /// key is term id, value is number of documents this token appears in
  std::unordered_map<TermID, uint32_t> documents_per_term_;

  /// key is document id, value is the vector that contains the values sparse representation of the
  /// tfidf values.
  std::unordered_map<DocumentID, std::vector<float>> document_to_vector_;

  /// key is document id, value is the vector that contains the TermIDs of the terms that this
  /// document contains
  std::unordered_map<DocumentID, std::vector<TermID>> document_to_contained_terms;

  std::unordered_map<std::string, TermID> term_to_term_id;

  /// TODO: Only needed for debugging and testing
  std::vector<std::string> term_id_to_term;
};

#endif  // VECTORSPACEMODELENGINE_HPP
