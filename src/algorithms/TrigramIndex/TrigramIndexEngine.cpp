//---------------------------------------------------------------------------
#include <stdexcept>
#include <cassert>
#include <string>
#include <iostream>
#include "TrigramIndexEngine.hpp"
#include "models/Trigram.hpp"
#include "parser/TrigramParser.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(DocumentIterator doc_it) {
    uint32_t doc_count = 0;
    uint64_t total_trigram_count = 0;
    std::unordered_map<int, uint32_t> doc_to_length;

    while(doc_it.hasNext()) {
        auto doc = *doc_it;
        uint32_t doc_length = 0;
        std::unordered_map<Trigram, uint32_t> appearances;

        const char* begin = doc->getBegin();
        const char* end = doc->getBegin() + doc->getSize();

        auto parser = TrigramParser(begin, end);
        while(parser.hasNext()) {
            ++appearances[parser.next()];
            ++doc_length;
        }

        // insert into index
        for(const auto& [trigram, count] : appearances) {
            index.insert(trigram, {doc->getId(), count});
        }

        // update statistics
        total_trigram_count += doc_length;
        doc_to_length[doc->getId()] = doc_length;
        ++doc_it;
        ++doc_count;
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