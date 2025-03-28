//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_TFIDF_HPP
#define VECTORSPACEMODELENGINE_TFIDF_HPP

#include <sys/types.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../documents/document_iterator.hpp"
#include "../../fts_engine.hpp"
#include "index/hnsw/hnsw_alg.hpp"
#include "index/hnsw/spaces/cosine_space_sparse.hpp"
#include "index/hnsw/spaces/l2_space.hpp"
#include "index/hnsw/spaces/l2_space_sparse.hpp"

namespace hnsw = vectorlib::hnsw;

using TermID = uint32_t;

class VectorEngineTfidf : public FullTextSearchEngine {
 public:
  VectorEngineTfidf();

  ~VectorEngineTfidf();
  void indexDocuments(std::string &data_path) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;
  void insert(DocumentID min_id, DocumentID max_id);
  uint64_t footprint() override;

 private:
  int dim = 1;                  // Dimension of the elements
  int max_elements = 10000000;  // Maximum number of elements, should be known beforehand.
  int M =
      15;  // the number of bi-directional links created for every new element during construction.
  int ef_construction = 20;  // Controls index search speed/build speed tradeoff

  hnsw::HierarchicalNSW<float> *hnsw_alg;

  // prints how often a token occurs in this vector
  void print_vector(std::vector<float> values, std::vector<TermID> terms);
  void store();

  /**
   * Stores the vector @documents_per_term with the following scheme:
   * [number_of_documents_that_contain_first_term][number_of_documents_that_contain_second_term][number_of_documents_that_contain_third_term]
   * total number of terms can be determined from the size of the file.
   * term_id can be determined from the location in the array.
   */
  void store_documents_per_term(std::string &file_name);

  /**
   * Stores the vector @document_to_vector_ with the following scheme:
   * [num_of_documents_in_total][size_of_first_vector][first_value_of_first_vector][second_value_of_first_vector]...[last_value_of_first_vector][size_of_second_vector]
   * All consecutive.
   */
  void store_document_to_vector(std::string &file_name);

  void store_document_to_contained_terms(std::string &file_name);
  /**
   * This function suffices. There is no need for a function to store term_to_term_id.
   */
  void store_term_id_to_term(std::string &file_name);

  bool load();
  void load_documents_per_term(std::string &file_name);
  void load_document_to_vector(std::string &file_name);
  void load_document_to_contained_terms(std::string &file_name);
  void load_term_id_to_term(std::string &file_name);

  /// key is term id, value is number of documents this token appears in
  // std::unordered_map<TermID, uint32_t> documents_per_term_;
  std::vector<uint32_t> documents_per_term_;

  /// key is document id, value is the vector that contains the values sparse representation of the
  /// tfidf values.
  // std::unordered_map<DocumentID, std::vector<float>> document_to_vector_;
  std::vector<std::vector<float>> document_to_vector_;

  /// key is document id, value is the vector that contains the TermIDs of the terms that this
  /// document contains
  // std::unordered_map<DocumentID, std::vector<TermID>> document_to_contained_terms_;
  std::vector<std::vector<TermID>> document_to_contained_terms_;

  std::unordered_map<std::string, TermID> term_to_term_id;

  /// Only needed for debugging and testing
  std::vector<std::string> term_id_to_term;

  hnsw::CosineSpaceSparse space;

  void normalize_vector(std::vector<float> &v);
};

#endif  // VECTORSPACEMODELENGINE_TFIDF_HPP
