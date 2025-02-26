// Created by fts on 10/31/24.

#include "trigram_index_engine.hpp"

#include <stdexcept>
#include <string>

void TrigramIndexEngine::indexDocuments(std::string &data_path) {
  throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<std::pair<DocumentID, double>> TrigramIndexEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  throw std::runtime_error("search method is not yet implemented.");
}

uint32_t TrigramIndexEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double TrigramIndexEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
