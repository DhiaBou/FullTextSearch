// Created by fts on 10/31/24.

#include "inverted_index_engine.hpp"

#include <stdexcept>
#include <string>

void InvertedIndexEngine::indexDocuments(DocumentIterator it) {
  throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<DocumentID> InvertedIndexEngine::search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func) {
  throw std::runtime_error("search method is not yet implemented.");
}

uint32_t InvertedIndexEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double InvertedIndexEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
