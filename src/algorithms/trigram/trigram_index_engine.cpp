#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
//---------------------------------------------------------------------------
#include "models/trigram.hpp"
#include "parser/trigram_parser.hpp"
#include "trigram_index_engine.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(DocumentIterator doc_it) {
  uint32_t doc_count = 0;
  uint64_t total_trigram_count = 0;
  std::unordered_map<int, uint32_t> doc_to_length;

  while (doc_it.hasNext()) {
    auto doc = *doc_it;
    uint32_t doc_length = 0;
    std::unordered_map<trigramlib::Trigram, uint32_t> appearances;

    const char* begin = doc->getData();
    const char* end = doc->getData() + doc->getSize();

    auto parser = trigramlib::TrigramParser(begin, end);
    while (parser.hasNext()) {
      ++appearances[parser.next()];
      ++doc_length;
    }

    // insert into index
    for (const auto& [trigram, count] : appearances) {
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
std::vector<uint32_t> TrigramIndexEngine::search(const std::string& query) {
  std::vector<trigramlib::Trigram> trigrams;

  auto parser = trigramlib::TrigramParser(query.c_str(), query.c_str() + query.size());
  while (parser.hasNext()) {
    trigrams.push_back(parser.next());
  }

  std::unordered_map<uint32_t, uint32_t> result;

  for (const auto& trigram : trigrams) {
    try {
      auto matches = index.lookup(trigram);
      for (const auto& match : *matches) {
        ++result[match.doc_id];
      }
    } catch (const std::out_of_range& e) {
      std::cerr << "Could not find Trigram: " << trigram << std::endl;
    }
  }

  std::vector<std::pair<uint32_t, uint32_t>> sorted_result(result.begin(), result.end());
  std::sort(sorted_result.begin(), sorted_result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  uint32_t count = 1;
  for (const auto& [doc, freq] : sorted_result) {
    std::cout << "Rank: " << count << std::endl;
    std::cout << "Document: " << doc << ", Freq: " << freq << std::endl;
    if (count++ == 10) break;
  }

  return std::vector<uint32_t>();
}