//
// Created by fts on 10/31/24.
//

#ifndef INVERTEDINDEXENGINE_HPP
#define INVERTEDINDEXENGINE_HPP

#include "../../FullTextSearchEngine.hpp"

class InvertedIndexEngine : public FullTextSearchEngine {
public:
    void indexDocuments(DocumentIterator doc_it) override;

    std::vector<std::shared_ptr<Document>> search(const std::string &query) override;

private:
    std::unordered_map<std::string, std::unordered_map<size_t, size_t>> index;
};

#endif  // INVERTEDINDEXENGINE_HPP

