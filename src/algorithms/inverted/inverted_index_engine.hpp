//
// Created by fts on 10/31/24.
//

#ifndef INVERTEDINDEXENGINE_HPP
#define INVERTEDINDEXENGINE_HPP

#include "../../fts_engine.hpp"

class InvertedIndexEngine : public FullTextSearchEngine {
public:
    void indexDocuments(DocumentIterator it) override;

    std::vector<std::shared_ptr<Document> > search(const std::string &query) override;

private:
};

#endif  // INVERTEDINDEXENGINE_HPP

