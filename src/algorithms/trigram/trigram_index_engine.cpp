// Created by fts on 10/31/24.

#include "trigram_index_engine.hpp"

#include <stdexcept>
#include <string>

void TrigramIndexEngine::indexDocuments(DocumentIterator it) {
  throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<uint32_t> TrigramIndexEngine::search(const std::string &query,
                                                 const scoring::ScoringFunction &score_func) {
  throw std::runtime_error("search method is not yet implemented.");
}

uint32_t TrigramIndexEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double TrigramIndexEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
