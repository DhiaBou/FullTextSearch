// Created by fts on 10/31/24.

#include "inverted_index_engine.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tokenizer/tokenizer.hpp"

void InvertedIndexEngine::indexDocuments(DocumentIterator doc_it) {
  do {
    auto doc = *doc_it;
    auto begin = doc->getData();
    auto end = begin + doc->getSize();

    Tokenizer tokenizer(begin, doc->getSize());

    while (tokenizer.hasMoreTokens()) {
      Token token = tokenizer.nextToken();

      if (!token.empty()) {
        // increment the number of times a token appeared in that document
        term_frequency_per_document_[token.getString()][doc->getId()]++;
        // increase the total number of terms in doc d
        tokens_per_document_[doc->getId()]++;
      }
    }

    ++doc_it;
  } while (doc_it.hasNext());
}

std::vector<DocumentID> InvertedIndexEngine::search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func) {
  // Tokenize the query
  Tokenizer tokenizer(query.c_str(), query.size());

  // Map of doc_id -> cumulative score
  std::unordered_map<size_t, double> doc_scores;

  // Compute scores for each token in the query
  while (tokenizer.hasMoreTokens()) {
    Token token = tokenizer.nextToken();
    if (!token.empty()) {
      auto it = term_frequency_per_document_.find(token.getString());
      if (it == term_frequency_per_document_.end()) {
        // This token doesn't appear in any document
        continue;
      }

      // For each document that contains this token, accumulate its score
      for (const auto &doc_freq : it->second) {
        size_t doc_id = doc_freq.first;
        double score = docScoreForToken(doc_id, token);
        doc_scores[doc_id] += score;
      }
    }
  }

  // Use a min-heap to track the top 10 documents by score
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>,
                      std::greater<>>
      results;

  for (const auto &entry : doc_scores) {
    const double &score = entry.second;
    const size_t &doc_id = entry.first;

    if (results.size() < 10) {
      results.emplace(score, doc_id);
    } else if (score > results.top().first) {
      results.pop();
      results.emplace(score, doc_id);
    }
  }

  // Extract top documents in descending order of score
  std::vector<uint32_t> top_documents(results.size());
  for (int i = static_cast<int>(results.size()) - 1; i >= 0; i--) {
    top_documents[i] = static_cast<uint32_t>(results.top().second);
    results.pop();
  }

  return top_documents;
}

double InvertedIndexEngine::docScoreForToken(size_t docId, const Token &token) {
  // Ensure token exists in the doc's frequency map, otherwise term frequency is 0.
  auto docFreqIt = term_frequency_per_document_.find(token.getString());
  if (docFreqIt == term_frequency_per_document_.end()) {
    return 0.0;
  }

  const auto &freqMap = docFreqIt->second;
  auto it = freqMap.find(docId);
  if (it == freqMap.end()) {
    return 0.0;  // token not in doc
  }

  size_t tf = it->second;
  size_t totalTokens = tokens_per_document_[docId];
  size_t docsContainingToken = freqMap.size();
  size_t totalDocs = tokens_per_document_.size();

  if (totalTokens == 0 || docsContainingToken == 0 || totalDocs == 0) {
    return 0.0;
  }

  float termFrequency = static_cast<float>(tf) / static_cast<float>(totalTokens);
  double idf = std::log((static_cast<double>(totalDocs) + 1.0) /
                        (static_cast<double>(docsContainingToken) + 1.0));

  return termFrequency * idf;
}

uint32_t InvertedIndexEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}

double InvertedIndexEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
