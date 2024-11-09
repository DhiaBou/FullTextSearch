//---------------------------------------------------------------------------
#include <stdexcept>
#include <string>
#include "TrigramIndexEngine.hpp"
#include "TrigramUtils.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(DocumentIterator it) {
    while(it.hasNext()) {
        auto doc = it.next();

        size_t size = doc->getSize();
        const char* begin = doc->getBegin();
        for(size_t i = 0; i < size-2; ++i, ++begin) {
            //TODO build trigrams
            auto trigram = Trigram(0);

            index.insert({trigram, static_cast<uint32_t>(doc->getId())});
        }
    }
}
//---------------------------------------------------------------------------
std::vector<std::shared_ptr<Document>> TrigramIndexEngine::search(const std::string &query) {
    throw std::runtime_error("search method is not yet implemented.");
}
//---------------------------------------------------------------------------