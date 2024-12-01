//---------------------------------------------------------------------------
#include <stdexcept>
#include <cassert>
#include <string>
#include <iostream>
#include "TrigramIndexEngine.hpp"
#include "models/Trigram.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(DocumentIterator doc_it) {
    uint32_t doc_count = 0;
    uint64_t total_trigram_count = 0;
    std::unordered_map<int, uint32_t> doc_to_length;

    while(doc_it.hasNext()) {
        auto doc = *doc_it;
        uint32_t doc_length = 0;
        std::unordered_map<Trigram, uint32_t> appearances;

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

                    auto offset = trigram_begin - word_begin;
                    ++appearances[{trigram_begin, static_cast<uint8_t>(offset)}];
                    ++doc_length;
                    
                    ++trigram_begin;
                }
            } else {
                if (it - word_begin == 2) {
                    // stand-alone two-character "trigram"
                    char trigram[3] = {*word_begin, *(word_begin+1), '\0'};
                    ++appearances[{trigram, 0}];
                    ++doc_length;
                }

                trigram_begin = it + 1;
                word_begin = it + 1;
            }
            ++it;
        }

        total_trigram_count += doc_length;
        doc_to_length[doc->getId()] = doc_length;
        
        for(const auto& pair : appearances) {
            index.insert(pair.first, {doc->getId(), pair.second});
        }
        
        ++doc_it;
    }

    auto avg_doc_length = total_trigram_count / doc_count;
}
//---------------------------------------------------------------------------
std::vector<std::shared_ptr<Document>> TrigramIndexEngine::search(const std::string &query) {
    try {
        auto matches = index.lookup({query.c_str(), 0});
        for(const auto& match : *matches) {
            std::cout << "Document: " << match.doc_id << ", Freq: " << match.freq << std::endl;
        }
    } catch (const std::out_of_range& e) {
        std::cerr << "Lookup error: " << e.what() << std::endl;
    }

    return std::vector<std::shared_ptr<Document>>();
}
//---------------------------------------------------------------------------