#include "vector_engine.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../scoring/tf_idf.hpp"
#include "../../tokenizer/simpletokenizer.hpp"

void VectorEngine::print_vector(DocumentID doc_id) {
  int i = 0;
  for (const auto &[token, num_docs_with_token] : documents_per_token_) {
    std::cout << "token index: " << i << ", token: " << token << "\n";
  }
}

void VectorEngine::indexDocuments(DocumentIterator doc_it) {
  // TODO:
  // 1. Count total frequency of a term in all documents all over
  // 2. Count frequency of a term per document
  // 3. Compute tf-idf vector for each document
  while (doc_it.hasNext()) {
    auto doc = *doc_it;
    tokenizer::SimpleTokenizer tokenizer(doc->getData(), doc->getSize());
    std::unordered_set<std::string> unique_tokens_in_doc;

    for (auto token = tokenizer.nextToken(false); !token.empty();
         token = tokenizer.nextToken(false)) {
      // increment the number of times a token appeared in that document
      unique_tokens_in_doc.insert(token);
      term_frequency_per_document_[token][doc->getId()]++;
      // increase the total number of terms in doc d
      tokens_per_document_[doc->getId()]++;
      // std::cout << token << "\n";
    }

    // std::cout << doc->getId() << "\n";
    // std::cout << tokens_per_document_[doc->getId()] << "\n";

    for (auto &t : unique_tokens_in_doc) {
      ++documents_per_token_[t];
    }

    // std::cout << documents_per_token_.size() << "\n";

    ++doc_it;
  }

  // todo: store compressed vectors by concatenating them and then using compression tool

  // TOdo: load indexed vectors from disc and decompress it and store it as vector of vectors

  // create the tf_idf vectors for each document
  uint32_t num_of_docs = tokens_per_document_.size();
  std::unique_ptr<scoring::ScoringFunction> score_func =
      std::make_unique<scoring::TfIdf>(num_of_docs);
  for (const auto &[id, num_tokens] : tokens_per_document_) {
    std::vector<uint32_t> vec;
    vec.reserve(documents_per_token_.size());
    for (const auto &[token, num_docs_with_token] : documents_per_token_) {
      vec.push_back(score_func->score(
          {num_tokens}, {term_frequency_per_document_[token][id], num_docs_with_token}));
    }
    document_to_vector_.emplace(id, vec);
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
