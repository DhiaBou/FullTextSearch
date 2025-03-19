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
#include "index/hnsw/hnsw_alg.hpp"
#include "index/hnsw/spaces/l2_space.hpp"
#include "index/hnsw/spaces/l2_space_doc_id.hpp"

namespace hnsw = vectorlib::hnsw;

using TermID = uint32_t;

class VectorEngine : public FullTextSearchEngine {
 public:
  VectorEngine();

  ~VectorEngine();
  void indexDocuments(DocumentIterator it) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
  int dim = 1;                  // Dimension of the elements
  int max_elements = 10000000;  // Maximum number of elements, should be known beforehand.
  // TODO: test different parameter combinations.
  int M =
      15;  // the number of bi-directional links created for every new element during construction.
  int ef_construction = 20;  // Controls index search speed/build speed tradeoff

  hnsw::HierarchicalNSW<float> *hnsw_alg;

  // prints how often a token occurs in this vector
  void print_vector(std::vector<float> values, std::vector<TermID> terms);
  // std::vector<float> compress_vector(std::vector<float> v);
  // std::vector<float> decompress_vector(std::vector<float> v);
  void store_vectors();
  void store_documents_per_term();
  void load_documents_per_term();
  void load_vectors();

  /// key is term id, value is number of documents this token appears in
  std::unordered_map<TermID, uint32_t> documents_per_term_;

  /// key is document id, value is the vector that contains the values sparse representation of the
  /// tfidf values.
  std::unordered_map<DocumentID, std::vector<float>> document_to_vector_;

  /// key is document id, value is the vector that contains the TermIDs of the terms that this
  /// document contains
  std::unordered_map<DocumentID, std::vector<TermID>> document_to_contained_terms_;

  std::unordered_map<std::string, TermID> term_to_term_id;

  /// TODO: Only needed for debugging and testing
  std::vector<std::string> term_id_to_term;

  hnsw::L2SpaceDocId space;
};

#endif  // VECTORSPACEMODELENGINE_HPP
