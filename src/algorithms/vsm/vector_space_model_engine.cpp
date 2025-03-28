// Created by fts on 10/31/24.

#include "vector_space_model_engine.hpp"

#include <stdexcept>
#include <string>

void VectorSpaceModelEngine::indexDocuments(std::string &data_path) {
  throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<std::pair<DocumentID, double>> VectorSpaceModelEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  throw std::runtime_error("search method is not yet implemented.");
}

uint64_t VectorSpaceModelEngine::footprint_capacity() {
  // TODO
  return 0;
}

uint64_t VectorSpaceModelEngine::footprint_size() {
  // TODO
  return 0;
}

uint32_t VectorSpaceModelEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double VectorSpaceModelEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
