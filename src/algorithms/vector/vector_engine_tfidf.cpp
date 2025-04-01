#include "vector_engine_tfidf.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "algorithms/vector/file_io.hpp"
#include "algorithms/vector/index/hnsw/spaces/cosine_space_sparse.hpp"
#include "fts_engine.hpp"
#include "scoring/tf_idf_gensim.hpp"
#include "tokenizer/stemmingtokenizer.hpp"

namespace hnsw = vectorlib::hnsw;

VectorEngineTfidf::VectorEngineTfidf()
    : space(dim, document_to_vector_, document_to_contained_terms_) {
  hnsw_alg = new hnsw::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
}

VectorEngineTfidf::~VectorEngineTfidf() { delete hnsw_alg; }

void VectorEngineTfidf::print_vector(std::vector<float> values,
                                     std::vector<TermID> contained_terms) {}

void VectorEngineTfidf::store_documents_per_term(std::string &file_name) {
  MmapedFileWriter dpt(file_name.c_str(), documents_per_term_.size() * sizeof(uint32_t));
  dpt.write(reinterpret_cast<const char *>(documents_per_term_.data()),
            documents_per_term_.size() * sizeof(uint32_t));
}

void VectorEngineTfidf::store_document_to_vector(std::string &file_name) {
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

    // write vector
    dtv.write(reinterpret_cast<const char *>(document_to_vector_[did].data()),
              document_to_vector_[did].size() * sizeof(float));
  }
}

void VectorEngineTfidf::store_document_to_contained_terms(std::string &file_name) {
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

void VectorEngineTfidf::store_term_id_to_term(std::string &file_name) {
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

void VectorEngineTfidf::store() {
  std::string dir_name = "stored_hnsw";
  std::string dpt_fn = dir_name + "/documents_per_term";
  std::string dtv_fn = dir_name + "/document_to_vector";
  std::string dtct_fn = dir_name + "/document_to_contained_terms";
  std::string titt_fn = dir_name + "/term_id_to_term";
  std::string hnsw_fn = dir_name + "/hnsw-graph";
  store_documents_per_term(dpt_fn);
  store_document_to_vector(dtv_fn);
  store_document_to_contained_terms(dtct_fn);
  store_term_id_to_term(titt_fn);
  hnsw_alg->saveIndex(hnsw_fn);
}

bool VectorEngineTfidf::load() {
  std::string dir_name = "stored_hnsw";
  std::string dpt_fn = dir_name + "/documents_per_term";
  std::string dtv_fn = dir_name + "/document_to_vector";
  std::string dtct_fn = dir_name + "/document_to_contained_terms";
  std::string titt_fn = dir_name + "/term_id_to_term";
  std::string hnsw_fn = dir_name + "/hnsw-graph";

  if (fs::exists(dir_name) && fs::is_directory(dir_name) && fs::exists(dpt_fn) &&
      fs::exists(dtv_fn) && fs::exists(dtct_fn) && fs::exists(titt_fn) && fs::exists(hnsw_fn)) {
    load_documents_per_term(dpt_fn);
    load_document_to_vector(dtv_fn);
    load_document_to_contained_terms(dtct_fn);
    load_term_id_to_term(titt_fn);
    hnsw_alg->loadIndex(hnsw_fn, static_cast<vectorlib::SpaceInterface<float> *>(&space),
                        max_elements);
    return true;
  } else {
    fs::remove_all(dir_name);
    fs::create_directory(dir_name);
    return false;
  }
}

void VectorEngineTfidf::load_documents_per_term(std::string &file_name) {
  MmapedFileReader dpt_reader(file_name);
  documents_per_term_.resize(dpt_reader.get_size() / sizeof(uint32_t));
  memcpy(documents_per_term_.data(), dpt_reader.begin(), dpt_reader.get_size() / sizeof(uint32_t));
}

void VectorEngineTfidf::load_document_to_vector(std::string &file_name) {
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

    // read in the actual vector
    memcpy(vec.data(), cur, vec.size() * sizeof(float));
    cur += vec.size() * sizeof(float);
  }
}

void VectorEngineTfidf::load_document_to_contained_terms(std::string &file_name) {
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

void VectorEngineTfidf::load_term_id_to_term(std::string &file_name) {
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

void VectorEngineTfidf::normalize_vector(std::vector<float> &v) {
  float sum = 0;
  for (int i = 0; i < v.size(); ++i) {
    sum += v[i] * v[i];
  }
  double norm = std::sqrt(sum);
  for (int i = 0; i < v.size(); ++i) {
    v[i] = v[i] / norm;
  }
}

void VectorEngineTfidf::indexDocuments(std::string &data_path) {
  DocumentIterator doc_it(data_path);
  if (load()) {
    std::cout << "HNSW index loaded from save file.\n";
    return;
  }
  // key is term id, value is a map of doc id to term frequency
  // needed only to build the vectors
  std::unordered_map<TermID, std::unordered_map<DocumentID, uint32_t>> term_frequency_per_document_;

  // key is document id, value is number of terms that this document contains
  // needed only to build the vectors
  std::unordered_map<DocumentID, uint32_t> terms_per_document_;
  std::vector<Document> docs = doc_it.next();
  while (!docs.empty()) {
    for (const auto &doc : docs) {
      DocumentID did = doc.getId() - 1;
      tokenizer::StemmingTokenizer tokenizer(doc.getData(), doc.getSize());
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
      }

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
    }
    docs = doc_it.next();
  }

  // create the tf_idf vectors for each document
  uint32_t num_of_docs = terms_per_document_.size();
  std::unique_ptr<scoring::ScoringFunction> score_func =
      std::make_unique<scoring::TfIdfGensim>(num_of_docs);

  // iterate through all documents
  for (DocumentID did = 0; did < document_to_contained_terms_.size(); ++did) {
    uint32_t num_tokens = terms_per_document_[did];
    std::vector<float> vec;

    vec.reserve(document_to_contained_terms_[did].size());

    // iterate over all terms in this document
    for (const auto tid : document_to_contained_terms_[did]) {
      vec.push_back((float)score_func->score(
          {num_tokens}, {term_frequency_per_document_[tid][did], documents_per_term_[tid]}));
    }
    normalize_vector(vec);

    document_to_vector_.push_back(std::move(vec));
  }

  // insert vectors into hnsw
  for (DocumentID doc_id = 0; doc_id < document_to_contained_terms_.size(); ++doc_id) {
    hnsw_alg->addPoint(&doc_id, doc_id);
  }

  store();
}

void VectorEngineTfidf::insert(DocumentID min_id_inclusive, DocumentID max_id_exclusive) {
  for (DocumentID doc_id = min_id_inclusive; doc_id < max_id_exclusive; ++doc_id) {
    hnsw_alg->addPoint(&doc_id, doc_id);
  }
}

std::vector<std::pair<DocumentID, double>> VectorEngineTfidf::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  DocumentID query_id = document_to_contained_terms_.size();

  std::unordered_map<TermID, uint32_t> query_term_frequency;
  std::unordered_set<TermID> unique_terms_in_query;
  uint32_t num_tokens_in_query = 0;

  tokenizer::StemmingTokenizer tokenizer(query.c_str(), query.size());

  for (auto token = tokenizer.nextToken(false); !token.empty();
       token = tokenizer.nextToken(false)) {
    ++num_tokens_in_query;
    if (term_to_term_id.find(token) == term_to_term_id.end()) {
      // if the term does not occour in any of the indexed documents, then the distance to every
      // document is the same in regard to this term, so we can just skip it.
      continue;
    }
    TermID tid = term_to_term_id[token];
    ++query_term_frequency[tid];
    unique_terms_in_query.insert(tid);
  }

  std::vector<uint32_t> sorted_tokens(unique_terms_in_query.begin(), unique_terms_in_query.end());
  std::sort(sorted_tokens.begin(), sorted_tokens.end());

  uint32_t num_of_docs = document_to_contained_terms_.size();

  std::vector<float> vec;
  vec.reserve(sorted_tokens.size());

  // iterate over all terms in this document
  for (const auto tid : sorted_tokens) {
    vec.push_back((float)score_func.score({num_tokens_in_query},
                                          {query_term_frequency[tid], documents_per_term_[tid]}));
  }

  normalize_vector(vec);

  // query data is only inserted for the time of the query.
  document_to_contained_terms_.push_back(std::move(sorted_tokens));
  document_to_vector_.push_back(std::move(vec));

  std::priority_queue<std::pair<float, size_t>> pq = hnsw_alg->searchKnn(&query_id, num_results);

  // remove query data
  document_to_contained_terms_.pop_back();
  document_to_vector_.pop_back();

  std::vector<std::pair<DocumentID, double>> res;
  res.resize(num_results);
  size_t i = num_results;
  while (!pq.empty()) {
    auto [dist, id] = pq.top();

    res[--i] = {id + 1, dist};
    pq.pop();
  }

  return res;
}

uint32_t VectorEngineTfidf::getDocumentCount() { return document_to_vector_.size(); }

double VectorEngineTfidf::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}

uint64_t VectorEngineTfidf::footprint_size() {
  uint64_t vec_fp = documents_per_term_.size() * sizeof(uint32_t);
  for (auto &v : document_to_vector_) {
    vec_fp += (v.size() * sizeof(float));
  }
  for (auto &v : document_to_contained_terms_) {
    vec_fp += (v.size() * sizeof(TermID));
  }
  for (auto &[s, t] : term_to_term_id) {
    vec_fp += 2 * s.size() + sizeof(uint32_t);  // count size of the string twice, because it is
                                                // contained in term_id_to_term as well
  }
  // uint64_t hnsw_fp = hnsw_alg->get_footprint();
  return vec_fp;
}

uint64_t VectorEngineTfidf::footprint_capacity() { return 0; }