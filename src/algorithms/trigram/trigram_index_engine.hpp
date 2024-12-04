//
// Created by fts on 10/31/24.
//

#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP

#include "../../fts_engine.hpp"

class TrigramIndexEngine : public FullTextSearchEngine {
public:
    void indexDocuments(DocumentIterator it) override;

    std::vector<std::shared_ptr<Document> > search(const std::string &query) override;

private:
};

#endif  // TRIGRAMINDEXENGINE_HPP
