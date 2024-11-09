//---------------------------------------------------------------------------
#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
#include "../../FullTextSearchEngine.hpp"
#include "TrigramIndex.hpp"
//---------------------------------------------------------------------------
class TrigramIndexEngine : public FullTextSearchEngine {
public:
    void indexDocuments(DocumentIterator it) override;

    std::vector<std::shared_ptr<Document>> search(const std::string &query) override;

private:
    TrigramIndex index;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
