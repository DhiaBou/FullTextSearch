// Created by fts on 10/31/24.

#include "vector_space_model_engine.hpp"

#include <stdexcept>
#include <string>

void VectorSpaceModelEngine::indexDocuments(DocumentIterator it) {
  throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<uint32_t> VectorSpaceModelEngine::search(const std::string &query,
                                                     const scoring::ScoringFunction &score_func) {
  throw std::runtime_error("search method is not yet implemented.");
}

uint32_t VectorSpaceModelEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double VectorSpaceModelEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
