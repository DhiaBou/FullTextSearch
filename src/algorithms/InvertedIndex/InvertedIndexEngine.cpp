// Created by fts on 10/31/24.

#include "InvertedIndexEngine.hpp"

#include <stdexcept>
#include <string>

void InvertedIndexEngine::indexDocuments(DocumentIterator it) {
    throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<std::shared_ptr<Document>> InvertedIndexEngine::search(const std::string &query) {
    throw std::runtime_error("search method is not yet implemented.");
}
