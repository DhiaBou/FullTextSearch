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
  uint64_t total_trigram_count = 0;

  while (doc_it.hasNext()) {
    uint32_t doc_length = 0;
    std::unordered_map<trigramlib::Trigram, uint32_t> appearances;

    auto doc = *doc_it;
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
    ++doc_count;

    ++doc_it;
  }

  avg_doc_length = static_cast<double>(total_trigram_count) / static_cast<double>(doc_count);
}
//---------------------------------------------------------------------------
std::vector<DocumentID> TrigramIndexEngine::search(const std::string& query,
                                                   const scoring::ScoringFunction& score_func) {
  const char* begin = query.c_str();
  const char* end = query.c_str() + query.size();
  trigramlib::TrigramParser parser(begin, end);

  std::unordered_map<DocumentID, double> doc_to_score;

  // Parse the query
  while (parser.hasNext()) {
    trigramlib::Trigram trigram = parser.next();

    // Lookup the trigram in the index
    std::vector<trigramlib::DocFreq>* matches = index.lookup(trigram);
    if (matches == nullptr) continue;

    for (const auto& match : *matches) {
      doc_to_score[match.doc_id] +=
          score_func.score({match.doc_id, doc_to_length[match.doc_id]},
                           {{{match.freq, static_cast<uint32_t>(matches->size())}}});
    }
  }

  // Order by score
  std::vector<std::pair<DocumentID, double>> sorted_result(doc_to_score.begin(),
                                                           doc_to_score.end());
  std::sort(sorted_result.begin(), sorted_result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Extract the top 10
  std::vector<DocumentID> top_10_results;
  for (size_t i = 0; i < 10 && i < sorted_result.size(); ++i) {
    top_10_results.push_back(sorted_result[i].first);
  }

  return top_10_results;
}
//---------------------------------------------------------------------------
uint32_t TrigramIndexEngine::getDocumentCount() { return doc_count; }
//---------------------------------------------------------------------------
double TrigramIndexEngine::getAvgDocumentLength() { return avg_doc_length; }