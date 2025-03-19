// Created by fts on 10/31/24.

#include "inverted_index_engine.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>

#include "../../documents/document_iterator.hpp"
#include "../../tokenizer/stemmingtokenizer.hpp"

void InvertedIndexEngine::indexDocuments(std::string &data_path) {
  DocumentIterator doc_it(data_path);

  auto index_batches = [&doc_it, this]() {
    std::vector<Document> current_batch = doc_it.next();
    while (!current_batch.empty()) {
      for (Document &doc : current_batch) {
        auto begin = doc.getData();

        tokenizer::StemmingTokenizer tokenizer(begin, doc.getSize());

        for (auto token = tokenizer.nextToken(false); !token.empty();
             token = tokenizer.nextToken(false)) {
          // increment the number of times a token appeared in that document
          auto increase_term_frequency = [&doc](std::unordered_map<uint32_t, uint32_t> &docs) {
            ++docs[doc.getId()];
          };
          term_frequency_per_document_.updateOrInsert(token, increase_term_frequency,
                                                      std::unordered_map<DocumentID, uint32_t>{});
          // increase the total number of terms in doc d
          auto increase_token_frequency = [](uint32_t &old_frequency) { old_frequency++; };
          tokens_per_document_.updateOrInsert(doc.getId(), increase_token_frequency, 0);
        }
      }
      current_batch = doc_it.next();
    }
  };

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 4;
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread{index_batches});
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

double InvertedIndexEngine::docScoreForToken(uint32_t docId, const std::string &token) {
  // Ensure token exists in the doc's frequency map, otherwise term frequency is 0.
  auto docFreqIt = term_frequency_per_document_.find(token);
  if (docFreqIt == term_frequency_per_document_.end()) {
    return 0.0;
  }

  const auto &freqMap = docFreqIt->second;
  auto it = freqMap.find(docId);
  if (it == freqMap.end()) {
    return 0.0;  // token not in doc
  }

  uint32_t tf = it->second;
  uint32_t totalTokens = *tokens_per_document_.get(docId);
  uint32_t docsContainingToken = freqMap.size();
  uint32_t totalDocs = 0;  // tokens_per_document_.size();

  if (totalTokens == 0 || docsContainingToken == 0 || totalDocs == 0) {
    return 0.0;
  }

  float termFrequency = static_cast<float>(tf) / static_cast<float>(totalTokens);
  double idf = std::log((static_cast<double>(totalDocs) + 1.0) /
                        (static_cast<double>(docsContainingToken) + 1.0));

  return termFrequency * idf;
}
std::vector<std::pair<DocumentID, double>> InvertedIndexEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  // Tokenize the query
  tokenizer::StemmingTokenizer tokenizer(query.c_str(), query.size());

  // Map of doc_id -> cumulative score
  std::unordered_map<uint32_t, double> doc_scores;

  // Compute scores for each token in the query
  for (auto token = tokenizer.nextToken(false); !token.empty();
       token = tokenizer.nextToken(false)) {
    auto it = term_frequency_per_document_.find(token);
    if (it == term_frequency_per_document_.end()) {
      // This token doesn't appear in any document
      continue;
    }

    // For each document that contains this token, accumulate its score
    for (const auto &doc_freq : it->second) {
      uint32_t doc_id = doc_freq.first;
      double score = docScoreForToken(doc_id, token);
      doc_scores[doc_id] += score;
    }
  }

  // Use a min-heap to track the top 10 documents by score
  std::priority_queue<std::pair<double, uint32_t>, std::vector<std::pair<double, uint32_t>>,
                      std::greater<>>
      results;

  for (const auto &entry : doc_scores) {
    const double &score = entry.second;
    const uint32_t &doc_id = entry.first;

    if (results.size() < num_results) {
      results.emplace(score, doc_id);
    } else if (score > results.top().first) {
      results.pop();
      results.emplace(score, doc_id);
    }
  }

  // Extract top documents in descending order of score
  std::vector<std::pair<DocumentID, double>> top_documents{results.size()};
  for (int i = static_cast<int>(results.size()) - 1; i >= 0; i--) {
    top_documents[i] = {static_cast<uint32_t>(results.top().second), results.top().first};
    results.pop();
  }

  return top_documents;
}

uint64_t InvertedIndexEngine::footprint() {
  // TODO
  return 0;
}

uint32_t InvertedIndexEngine::getDocumentCount() { return tokens_per_document_.size(); }

double InvertedIndexEngine::getAvgDocumentLength() {
  if (average_doc_length_ != -1) {
    return average_doc_length_;
  }
  average_doc_length_ =
      true  // tokens_per_document_.empty()
          ? 0.0
          : static_cast<double>(std::accumulate(
                tokens_per_document_.begin(), tokens_per_document_.end(), 0ull,
                [](uint32_t sum, const auto &entry) { return sum + entry.second; })) /
                static_cast<double>(tokens_per_document_.size());
  return average_doc_length_;
}