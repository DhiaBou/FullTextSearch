#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
//---------------------------------------------------------------------------
#include "../../tokenizer/simpletokenizer.hpp"
#include "models/trigram.hpp"
#include "parser/trigram_parser.hpp"
#include "trigram_index_engine.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(DocumentIterator doc_it) {
  uint64_t total_trigram_count = 0;

  // Iterate documents
  while (doc_it.hasNext()) {
    uint32_t doc_length = 0;
    std::unordered_map<trigramlib::Trigram, uint32_t> appearances;
    auto doc = *doc_it;

    tokenizer::SimpleTokenizer tokenizer(doc->getData(), doc->getSize());

    // Tokenize the document
    for (auto token = tokenizer.nextToken(false); !token.empty();
         token = tokenizer.nextToken(false)) {
      const char* begin = token.c_str();
      const char* end = token.c_str() + token.size();
      trigramlib::TrigramParser trigram_parser(begin, end);

      // Parse the token
      while (trigram_parser.hasNext()) {
        ++appearances[trigram_parser.next()];
        ++doc_length;
      }
    }

    // Insert into index
    for (const auto& [trigram, count] : appearances) {
      index.insert(trigram, {doc->getId(), count});
    }

    // Update statistics
    total_trigram_count += doc_length;
    doc_to_length[doc->getId()] = doc_length;
    ++doc_count;

    ++doc_it;
  }

  avg_doc_length = static_cast<double>(total_trigram_count) / static_cast<double>(doc_count);
}
//---------------------------------------------------------------------------
std::vector<std::pair<DocumentID, double>> TrigramIndexEngine::search(
    const std::string& query, const scoring::ScoringFunction& score_func, uint32_t num_results) {
  std::unordered_map<DocumentID, double> doc_to_score;
  tokenizer::SimpleTokenizer tokenizer(query.c_str(), query.size());

  // Tokenize the query
  for (auto token = tokenizer.nextToken(false); !token.empty();
       token = tokenizer.nextToken(false)) {
    const char* begin = token.c_str();
    const char* end = token.c_str() + token.size();
    trigramlib::TrigramParser trigram_parser(begin, end);

    // The query's trigrams found in the index
    std::vector<std::vector<trigramlib::DocFreq>*> trigram_results;

    // Parse the token
    while (trigram_parser.hasNext()) {
      trigramlib::Trigram trigram = trigram_parser.next();

      // Lookup the trigram in the index
      trigram_results.emplace_back(index.lookup(trigram));
    }

    // Aggregate and normalize the scores
    for (const auto& result : trigram_results) {
      if (result == nullptr) continue;

      for (const auto& match : *result) {
        doc_to_score[match.doc_id] +=
            score_func.score({doc_to_length[match.doc_id]},
                             {match.freq, static_cast<uint32_t>(result->size())}) /
            static_cast<double>(trigram_results.size());
      }
    }
  }

  // Order by score
  std::vector<std::pair<DocumentID, double>> ordered_docs(doc_to_score.begin(), doc_to_score.end());
  std::sort(ordered_docs.begin(), ordered_docs.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Extract top results
  std::vector<std::pair<DocumentID, double>> results;
  for (size_t i = 0; i < num_results && i < ordered_docs.size(); ++i) {
    results.push_back(ordered_docs[i]);
  }

  return results;
}
//---------------------------------------------------------------------------
uint32_t TrigramIndexEngine::getDocumentCount() { return doc_count; }
//---------------------------------------------------------------------------
double TrigramIndexEngine::getAvgDocumentLength() { return avg_doc_length; }