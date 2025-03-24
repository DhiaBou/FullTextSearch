#include "vector_engine.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
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
  std::string file_name = "documents_per_term";
  MmapedFileWriter dpt(file_name.c_str(), documents_per_term_.size() * sizeof(uint32_t));
  dpt.write(reinterpret_cast<const char *>(documents_per_term_.data()),
            documents_per_term_.size() * sizeof(uint32_t));
}

void VectorEngine::store_document_to_vector() {
  std::string file_name = "document_to_vector";

  // just take an approximate size as initial max size. It will be scaled up or down if needed.
  MmapedFileWriter dtv(file_name.c_str(),
                       document_to_vector_.size() * document_to_vector_[0].size() * sizeof(float));

  // write total number of documents
  uint32_t num_docs = static_cast<uint32_t>(document_to_vector_.size());
  dtv.write(&num_docs, sizeof(uint32_t));

  for (DocumentID did = 0; did < document_to_vector_.size(); ++did) {
    // write size of vector
    uint32_t vector_size = static_cast<uint32_t>(document_to_vector_[did].size());
    dtv.write(&vector_size, sizeof(uint32_t));

    // std::cout << document_to_vector_[did][0] << "\n";

    // write vector
    dtv.write(reinterpret_cast<const char *>(document_to_vector_[did].data()),
              document_to_vector_[did].size() * sizeof(float));
  }
}

void VectorEngine::store_document_to_contained_terms() {
  std::string file_name = "document_to_contained_terms";

  // just take an approximate size as initial max size. It will be scaled up or down if needed.
  MmapedFileWriter dtct(file_name.c_str(), document_to_contained_terms_.size() *
                                               document_to_contained_terms_[0].size() *
                                               sizeof(TermID));

  // write total number of documents
  uint32_t num_docs = static_cast<uint32_t>(document_to_contained_terms_.size());
  dtct.write(&num_docs, sizeof(uint32_t));

  for (DocumentID did = 0; did < document_to_contained_terms_.size(); ++did) {
    // write size of vector
    uint32_t vector_size = static_cast<uint32_t>(document_to_contained_terms_[did].size());
    dtct.write(&vector_size, sizeof(uint32_t));

    // write vector
    dtct.write(reinterpret_cast<const char *>(document_to_contained_terms_[did].data()),
               document_to_contained_terms_[did].size() * sizeof(TermID));
  }
}

void VectorEngine::store_term_id_to_term() {
  std::string file_name = "term_id_to_term";

  // just take an approximate size as initial max size. It will be scaled up or down if needed.
  MmapedFileWriter titt(file_name.c_str(),
                        term_id_to_term.size() * term_id_to_term[0].size() * sizeof(char));

  // write total number of terms
  uint32_t num_docs = static_cast<uint32_t>(term_id_to_term.size());
  titt.write(&num_docs, sizeof(uint32_t));

  // write every string
  for (uint32_t i = 0; i < term_id_to_term.size(); ++i) {
    // copy the string including a null terminator at the end
    titt.write(term_id_to_term[i].data(), term_id_to_term[i].size() + 1);
  }
}

void VectorEngine::store_vectors() {
  store_documents_per_term();
  store_document_to_vector();
  store_document_to_contained_terms();
  store_term_id_to_term();

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
  std::string file_name = "documents_per_term";
  MmapedFileReader dpt_reader(file_name);
  documents_per_term_.reserve(dpt_reader.get_size() / sizeof(uint32_t));
  memcpy(documents_per_term_.data(), dpt_reader.begin(), dpt_reader.get_size() / sizeof(uint32_t));
}

void VectorEngine::load_document_to_vector() {
  std::string file_name = "document_to_vector";
  MmapedFileReader dtv_reader(file_name);
  const char *cur = dtv_reader.begin();

  // read in total number of documents in document_to_vector
  document_to_vector_.resize(*reinterpret_cast<const uint32_t *>(cur));
  cur += sizeof(uint32_t);

  for (DocumentID did = 0; did < document_to_vector_.size(); ++did) {
    std::vector<float> &vec = document_to_vector_[did];

    // read in the size of the vector for this document
    vec.resize(*reinterpret_cast<const uint32_t *>(cur));
    cur += sizeof(uint32_t);

    // std::cout << *reinterpret_cast<const float *>(cur) << "\n";

    // read in the actual vector
    memcpy(vec.data(), cur, vec.size() * sizeof(float));
    cur += vec.size() * sizeof(float);
  }
}

void VectorEngine::load_document_to_contained_terms() {
  std::string file_name = "document_to_contained_terms";
  MmapedFileReader dtct_reader(file_name);
  const char *cur = dtct_reader.begin();

  // read in total number of documents in document_to_vector
  document_to_contained_terms_.resize(*reinterpret_cast<const uint32_t *>(cur));
  cur += sizeof(uint32_t);

  for (DocumentID did = 0; did < document_to_vector_.size(); ++did) {
    std::vector<TermID> &vec = document_to_contained_terms_[did];

    // read in the size of the vector for this document
    vec.resize(*reinterpret_cast<const uint32_t *>(cur));
    cur += sizeof(uint32_t);

    // read in the actual vector
    memcpy(vec.data(), cur, vec.size() * sizeof(float));
    cur += vec.size() * sizeof(TermID);
  }
}

void VectorEngine::load_term_id_to_term() {
  std::string file_name = "term_id_to_term";
  MmapedFileReader titt_reader(file_name);
  const char *cur = titt_reader.begin();

  // read in total number of documents in document_to_vector
  uint32_t count = *reinterpret_cast<const uint32_t *>(cur);
  term_id_to_term.reserve(count);
  term_to_term_id.reserve(count);
  cur += sizeof(uint32_t);

  for (uint32_t i = 0; i < count; ++i) {
    std::string str(cur);   // constructs from null-terminated string
    cur += str.size() + 1;  // move past this string and the null terminator
    term_id_to_term.emplace_back(std::move(str));
    term_to_term_id[term_id_to_term[i]] = i;
  }
}

void VectorEngine::load_vectors() {
  load_documents_per_term();
  load_document_to_vector();
  load_document_to_contained_terms();
  load_term_id_to_term();
}

void VectorEngine::test_store_and_load() {
  store_vectors();
  std::cout << "documents_per_term now contains " << documents_per_term_.size() << " values.\n";
  for (TermID tid = 0; tid < 100; ++tid) {
    std::cout << documents_per_term_[tid] << " ";
  }
  std::cout << std::endl;
  documents_per_term_.clear();
  std::cout << "documents_per_term now contains " << documents_per_term_.size() << " values.\n";
  std::cout << "document_to_vector now contains " << document_to_vector_.size() << " values.\n";
  for (auto e : document_to_vector_[1]) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
  document_to_vector_.clear();
  std::cout << "document_to_vector now contains " << document_to_vector_.size() << " values.\n";
  std::cout << "document_to_contained_terms now contains " << document_to_contained_terms_.size()
            << " values.\n";
  for (auto e : document_to_contained_terms_[1]) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
  document_to_contained_terms_.clear();
  std::cout << "document_to_contained_terms now contains " << document_to_contained_terms_.size()
            << " values.\n";

  for (int i = 0; i < 10; ++i) {
    std::cout << term_id_to_term[i] << " ";
    std::cout << term_to_term_id[term_id_to_term[i]] << " ";
  }
  std::cout << std::endl;
  term_id_to_term.clear();
  term_to_term_id.clear();

  load_vectors();
  std::cout << "documents_per_term now contains " << documents_per_term_.size() << " values.\n";
  for (TermID tid = 0; tid < 100; ++tid) {
    std::cout << documents_per_term_[tid] << " ";
  }
  std::cout << std::endl;

  std::cout << "document_to_vector now contains " << document_to_vector_.size() << " values.\n";
  for (auto e : document_to_vector_[1]) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
  std::cout << "document_to_contained_terms now contains " << document_to_contained_terms_.size()
            << " values.\n";
  for (auto e : document_to_contained_terms_[1]) {
    std::cout << e << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 10; ++i) {
    std::cout << term_id_to_term[i] << " ";
    std::cout << term_to_term_id[term_id_to_term[i]] << " ";
  }
  std::cout << std::endl;
}

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
    // std::cout << "did: " << did << "\n";

    // std::cout << "document_to_contained_terms_[did].size(): "
    //           << document_to_contained_terms_[did].size() << "\n";
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

  test_store_and_load();

  // insert vectors into hnsw
  for (DocumentID doc_id = 0; doc_id < document_to_contained_terms_.size(); ++doc_id) {
    hnsw_alg->addPoint(&doc_id, doc_id);
    // if (doc_id % 10000 == 0) {
    std::cout << "inserted: " << doc_id << "\n";
    // }
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
