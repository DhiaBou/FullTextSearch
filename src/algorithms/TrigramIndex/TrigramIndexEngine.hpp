//---------------------------------------------------------------------------
#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
#include <unordered_set>
#include "../../FullTextSearchEngine.hpp"
#include "TrigramIndex.hpp"
#include "TrigramUtils.hpp"
//---------------------------------------------------------------------------
class TrigramIndexEngine : public FullTextSearchEngine {
public:
    /// Build the index.
    void indexDocuments(DocumentIterator it) override;
    /// Search for string.
    std::vector<std::shared_ptr<Document>> search(const std::string &query) override;
private:
    /// A whitelist of ASCII characters allowed in the trigrams.
    static constexpr std::array<bool, 128> white_list = utils::generateWhitelist();
    /// The underlying index structure.
    TrigramIndex index;
    /// The trigrams.
    std::unordered_set<Trigram> trigrams;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAMINDEXENGINE_HPP
//---------------------------------------------------------------------------
