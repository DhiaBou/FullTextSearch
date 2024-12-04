// Created by fts on 10/31/24.

#include "trigram_index_engine.hpp"

#include <stdexcept>
#include <string>

void TrigramIndexEngine::indexDocuments(DocumentIterator it) {
    throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<std::shared_ptr<Document> > TrigramIndexEngine::search(const std::string &query) {
    throw std::runtime_error("search method is not yet implemented.");
}
