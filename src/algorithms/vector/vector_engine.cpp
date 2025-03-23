#include "vector_engine.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../scoring/tf_idf.hpp"
#include "../../tokenizer/stemmingtokenizer.hpp"
#include "./file_io.hpp"
#include "algorithms/vector/index/hnsw/spaces/l2_space.hpp"
#include "fts_engine.hpp"

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

void VectorEngine::store_documents_per_term() {
  std::vector<uint32_t> v;
  v.reserve(documents_per_term_.size());
  for (const auto &num_docs : documents_per_term_) {
    v.push_back(num_docs);
  }
  std::string documents_per_term_fname = "documents_per_term";
  MmapedFileWriter dpt(documents_per_term_fname.c_str(), v.size() * sizeof(uint32_t));
  dpt.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(uint32_t));
}

void VectorEngine::store_vectors() {
  store_documents_per_term();

  // TODO: for the other vectors, store the size directly after the id, then you can just memcpy it
  // completely.
  // TODO: since TermID and DocumentID are consecutive, don't use a hashmap, but just a vector.
  // Preallocate their sizes.

  // size_t initial_max_size = 1024L * 1024;
  // std::string document_to_vector_fname = "document_to_vector.csv";
  // std::string document_to_contained_terms_fname = "document_to_contained_terms.csv";
  // std::string term_to_term_id_fname = "term_to_term_id.csv";
  // std::string term_id_to_term_fname = "term_id_to_term.csv";

  // // create mmaped files
  // MmapedFile dtv(document_to_vector_fname.c_str(), initial_max_size);
  // MmapedFile dtct(document_to_contained_terms_fname.c_str(), initial_max_size);
  // MmapedFile ttti(term_to_term_id_fname.c_str(), initial_max_size);
  // MmapedFile titt(term_id_to_term_fname.c_str(), initial_max_size);
}

void VectorEngine::load_documents_per_term() {
  std::string dpt_fname = "documents_per_term";
  MmapedFileReader dpt_reader(dpt_fname);
  documents_per_term_.reserve(dpt_reader.get_size() / sizeof(uint32_t));
  memcpy(documents_per_term_.data(), dpt_reader.begin(), dpt_reader.get_size() / sizeof(uint32_t));
}
void VectorEngine::load_vectors() { load_documents_per_term(); }

void VectorEngine::indexDocuments(DocumentIterator doc_it) {
  // key is term id, value is a map of doc id to term frequency
  // needed only to build the vectors
  std::unordered_map<TermID, std::unordered_map<DocumentID, uint32_t>> term_frequency_per_document_;

  // key is document id, value is number of terms that this document contains
  // needed only to build the vectors
  std::unordered_map<DocumentID, uint32_t> terms_per_document_;

  while (doc_it.hasNext()) {
    auto doc = *doc_it;
    DocumentID did = doc->getId() - 1;
    tokenizer::StemmingTokenizer tokenizer(doc->getData(), doc->getSize());
    std::unordered_set<TermID> unique_terms_in_doc;

    for (auto token = tokenizer.nextToken(false); !token.empty();
         token = tokenizer.nextToken(false)) {
      // if this is the first time this term was discoverd over all documents, give it a new
      // TermID
      if (term_to_term_id.find(token) == term_to_term_id.end()) {
        term_id_to_term.push_back(token);
        documents_per_term_.push_back(0);  // because it is increased later
        term_to_term_id[token] = term_id_to_term.size() - 1;
      }
      TermID tid = term_to_term_id[token];
      unique_terms_in_doc.insert(tid);

      // increment the number of times a token appeared in that document
      term_frequency_per_document_[tid][did]++;

      // increase the total number of terms in this document
      terms_per_document_[did]++;

      // std::cout << token << "\n";
    }

    // if (did % 10000 == 0) {
    std::cout << did << "\n";
    // }

    // for every term that occurs in this document, increase the count of documents that this term
    // occurs in
    for (auto &t : unique_terms_in_doc) {
      ++documents_per_term_[t];
    }

    // store the terms that this document contains as part of the sparse representation of the
    // tfidf vectors
    std::vector<uint32_t> sorted_tokens(unique_terms_in_doc.begin(), unique_terms_in_doc.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end());

    document_to_contained_terms_.push_back(std::move(sorted_tokens));

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
  for (DocumentID did = 0; did < document_to_contained_terms_.size(); ++did) {
    uint32_t num_tokens = terms_per_document_[did];
    std::vector<float> vec;
    std::cout << "did: " << did << "\n";

    std::cout << "document_to_contained_terms_[did].size(): "
              << document_to_contained_terms_[did].size() << "\n";
    vec.reserve(document_to_contained_terms_[did].size());

    // iterate over all terms in this document
    for (const auto tid : document_to_contained_terms_[did]) {
      vec.push_back((float)score_func->score(
          {num_tokens}, {term_frequency_per_document_[tid][did], documents_per_term_[tid]}));
    }

    document_to_vector_.push_back(std::move(vec));

    // if (debug_counter % 10000 == 0) {
    std::cout << debug_counter << "\n";
    // }
    ++debug_counter;
  }
  print_vector(document_to_vector_[2], document_to_contained_terms_[1]);

  // TODO: test storing vectors
  store_vectors();
  for (TermID tid = 0; tid < 100; ++tid) {
    std::cout << documents_per_term_[tid] << " ";
  }
  std::cout << "\n";
  documents_per_term_.clear();
  std::cout << "documents_per_term now contains " << documents_per_term_.size() << " values.\n";
  load_vectors();
  for (TermID tid = 0; tid < 100; ++tid) {
    std::cout << documents_per_term_[tid] << " ";
  }

  // insert vectors into hnsw
  for (DocumentID doc_id = 0; doc_id < document_to_contained_terms_.size(); ++doc_id) {
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
