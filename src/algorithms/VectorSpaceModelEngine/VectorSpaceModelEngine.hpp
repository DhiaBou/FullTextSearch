//
// Created by dhia on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include "../../FullTextSearchEngine.hpp"

class VectorSpaceModelEngine : public FullTextSearchEngine {
   public:
    void indexDocuments(const std::vector<Document>& documents) override;
    std::vector<Document> search(const std::string& query) override;

   private:
    // Term-document matrix, vocabulary, and other necessary data structures
};

#endif  // VECTORSPACEMODELENGINE_HPP
