#include "vector_engine.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../scoring/tf_idf.hpp"
#include "../../tokenizer/stemmingtokenizer.hpp"
#include "./file_io.hpp"

namespace hnsw = vectorlib::hnsw;

VectorEngine::VectorEngine() : space(dim, document_to_vector_, document_to_contained_terms_) {
  hnsw_alg = new hnsw::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
  // std::cout << "Vector Engine Initialized" << std::endl;
}

VectorEngine::~VectorEngine() {
  // hnsw_alg->saveIndex("hnswIndex.bin");
  delete hnsw_alg;
}

void VectorEngine::print_vector(std::vector<float> values, std::vector<TermID> contained_terms) {
  for (int i = 0; i < contained_terms.size(); ++i) {
    std::cout << term_id_to_term[contained_terms[i]] << ": " << values[i] << "; ";
  }
  std::cout << "\n";
}

// std::vector<float> VectorEngine::decompress_vector(std::vector<float> v) {}

void VectorEngine::store_vectors() {}

void VectorEngine::load_vectors() {}

void VectorEngine::indexDocuments(DocumentIterator doc_it) {
  // key is term id, value is a map of doc id to term frequency
  // needed only to build the vectors
  std::unordered_map<TermID, std::unordered_map<DocumentID, uint32_t>> term_frequency_per_document_;

  // key is document id, value is number of terms that this document contains
  // needed only to build the vectors
  std::unordered_map<DocumentID, uint32_t> terms_per_document_;

  while (doc_it.hasNext()) {
    auto doc = *doc_it;
    tokenizer::StemmingTokenizer tokenizer(doc->getData(), doc->getSize());
    std::unordered_set<TermID> unique_terms_in_doc;

    for (auto token = tokenizer.nextToken(false); !token.empty();
         token = tokenizer.nextToken(false)) {
      // if this is the first time this term was discoverd over all documents, give it a new TermID
      if (term_to_term_id.find(token) == term_to_term_id.end()) {
        term_id_to_term.push_back(token);
        term_to_term_id[token] = term_id_to_term.size() - 1;
      }
      TermID tid = term_to_term_id[token];
      unique_terms_in_doc.insert(tid);

      // increment the number of times a token appeared in that document
      term_frequency_per_document_[tid][doc->getId()]++;

      // increase the total number of terms in this document
      terms_per_document_[doc->getId()]++;

      // std::cout << token << "\n";
    }

    if (doc->getId() % 10000 == 0) {
      std::cout << doc->getId() << "\n";
    }
    // std::cout << tokens_per_document_[doc->getId()] << "\n";

    // for every term that occurs in this document, increase the count of documents that this term
    // occurs in
    for (auto &t : unique_terms_in_doc) {
      ++documents_per_term_[t];
    }

    // store the terms that this document contains as part of the sparse representation of the tfidf
    // vectors
    std::vector<uint32_t> sorted_tokens(unique_terms_in_doc.begin(), unique_terms_in_doc.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end());

    document_to_contained_terms_[doc->getId()] = std::move(sorted_tokens);

    // std::cout << documents_per_token_.size() << "\n";

    ++doc_it;
  }

  // todo: store compressed vectors by concatenating them and then using compression tool

  // TOdo: load indexed vectors from disc and decompress it and store it as vector of vectors

  // create the tf_idf vectors for each document
  uint32_t num_of_docs = terms_per_document_.size();
  std::unique_ptr<scoring::ScoringFunction> score_func =
      std::make_unique<scoring::TfIdf>(num_of_docs);

  // iterate through all documents
  uint32_t debug_counter = 0;
  for (const auto &[doc_id, num_tokens] : terms_per_document_) {
    std::vector<float> vec;
    vec.reserve(document_to_contained_terms_[doc_id].size());

    // iterate over all terms in this document
    for (const auto tid : document_to_contained_terms_[doc_id]) {
      vec.push_back((float)score_func->score(
          {num_tokens}, {term_frequency_per_document_[tid][doc_id], documents_per_term_[tid]}));
    }

    document_to_vector_[doc_id] = std::move(vec);

    if (debug_counter % 10000 == 0) {
      std::cout << debug_counter << "\n";
    }
    ++debug_counter;
  }
  print_vector(document_to_vector_[2], document_to_contained_terms_[2]);

  // insert vectors into hnsw
  for (const auto &[doc_id, _] : document_to_contained_terms_) {
    hnsw_alg->addPoint(&doc_id, doc_id);

    if (doc_id % 10000 == 0) std::cout << "inserted: " << doc_id << "\n";
  }
}

std::vector<std::pair<DocumentID, double>> VectorEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  throw std::runtime_error("search method is not yet implemented.");
}

uint32_t VectorEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double VectorEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
