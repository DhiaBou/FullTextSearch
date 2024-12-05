//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include "../../FullTextSearchEngine.hpp"

class VectorSpaceModelEngine : public FullTextSearchEngine {
public:

    VectorSpaceModelEngine();

    void indexDocuments(DocumentIterator it) override;

    std::vector<std::shared_ptr<Document> > search(const std::string &query) override;

private:
};

#endif  // VECTORSPACEMODELENGINE_HPP
