//---------------------------------------------------------------------------
#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
#include <unordered_set>
#include "../../FullTextSearchEngine.hpp"
#include "index/HashIndex.hpp"
#include "models/Trigram.hpp"
//---------------------------------------------------------------------------
class TrigramIndexEngine : public FullTextSearchEngine {
public:
    /// Build the index.
    void indexDocuments(DocumentIterator it) override;
    /// Search for string.
    std::vector<std::shared_ptr<Document>> search(const std::string &query) override;
private:
    /// The underlying index.
    HashIndex<16> index;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
