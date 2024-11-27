//---------------------------------------------------------------------------
#include <stdexcept>
#include <cassert>
#include <string>
#include <iostream>
#include "TrigramIndexEngine.hpp"
#include "Trigram.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(DocumentIterator doc_it) {
    uint32_t doc_count = 0;
    uint64_t total_trigram_count = 0;
    std::unordered_map<int, uint32_t> doc_to_length;

    while(doc_it.hasNext()) {
        auto doc = *doc_it;
        uint32_t doc_length = 0;
        ++doc_count;

        auto trigram_begin = doc->getBegin();
        auto word_begin = doc->getBegin();
        auto end = doc->getBegin() + doc->getSize();

        auto it = doc->getBegin();
        while(it < end) {
            auto c = static_cast<unsigned char>(*it);
            if(c < 128 && white_list[c]) {
                // white-listed ASCII character
                if(it - trigram_begin >= 2) {
                    // we have collected three consecutive trigrams
                    assert(it - trigram_begin == 2);
                    assert(trigram_begin >= word_begin);

                    auto offset = static_cast<uint8_t>(trigram_begin - word_begin);
                    trigrams.insert({trigram_begin, offset});
                    ++doc_length;
                    
                    ++trigram_begin;
                }
            } else {
                if (it - word_begin == 2) {
                    // stand-alone two-character "trigram"
                    char trigram[3] = {*word_begin, *(word_begin+1), '\0'};
                    trigrams.emplace(trigram, 0);
                    ++doc_length;
                }

                trigram_begin = it + 1;
                word_begin = it + 1;
            }
            ++it;
        }

        total_trigram_count += doc_length;
        doc_to_length[doc->getId()] = doc_length;
        
        ++doc_it;
    }

    auto avg_doc_length = total_trigram_count / doc_count;
}
//---------------------------------------------------------------------------
std::vector<std::shared_ptr<Document>> TrigramIndexEngine::search(const std::string &query) {
    std::cout << trigrams.size() << std::endl;
    throw std::runtime_error("search method is not yet implemented.");
}
//---------------------------------------------------------------------------