// Created by fts on 10/31/24.

#include "inverted_index_engine.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>

#include "datastructures/hyperloglog.hpp"
#include "documents/document_iterator.hpp"
#include "tokenizer/simpletokenizer.hpp"
#include "tokenizer/stemmingtokenizer.hpp"

void InvertedIndexEngine::indexDocuments(std::string &data_path) {

  auto [num_token, max_doc_id] = getNumTokensAndMaxDocId(data_path);
  term_frequency_per_document_ = ParallelHashTable<std::string, std::unordered_map<DocumentID, uint32_t>>{num_token};
  tokens_per_document_.resize(max_doc_id + 1);

  DocumentIterator doc_it(data_path);

  auto index_batches = [&doc_it, this]() {
    std::vector<Document> current_batch = doc_it.next();
    while (!current_batch.empty()) {
      for (Document &doc : current_batch) {
        auto begin = doc.getData();

        tokenizer::StemmingTokenizer tokenizer(begin, doc.getSize());
        // tokenizer::SimpleTokenizer tokenizer(begin, doc.getSize());

        for (auto token = tokenizer.nextToken(false); !token.empty();
             token = tokenizer.nextToken(false)) {
          // increment the number of times a token appeared in that document
          auto increase_term_frequency = [&doc](std::unordered_map<uint32_t, uint32_t> &docs) {
            ++docs[doc.getId()];
          };
          term_frequency_per_document_.updateOrInsert(token, increase_term_frequency,
                                                      std::unordered_map<DocumentID, uint32_t>{});
          // increase the total number of terms in doc d
          tokens_per_document_[doc.getId()]++;
        }
      }
      current_batch = doc_it.next();
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; i++) {
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
  uint32_t totalTokens = tokens_per_document_[docId];
  uint32_t docsContainingToken = freqMap.size();
  uint32_t totalDocs = tokens_per_document_.size();

  if (totalTokens == 0 || docsContainingToken == 0 || totalDocs == 0) {
    return 0.0;
  }

  float termFrequency = static_cast<float>(tf) / static_cast<float>(totalTokens);
  double idf = std::log((static_cast<double>(totalDocs) + 1.0) /
                        (static_cast<double>(docsContainingToken) + 1.0));

  return termFrequency * idf;
}

std::pair<uint64_t, uint64_t> InvertedIndexEngine::getNumTokensAndMaxDocId(const std::string &data_path) const {
  DocumentIterator doc_it(data_path);

  std::atomic<uint32_t> max_doc_id = 0;

  HyperLogLog<std::string> hyper_log_log{NUM_THREADS};

  auto count_distinct_tokens = [&doc_it, &hyper_log_log, &max_doc_id](uint64_t thread_id) {
    std::vector<Document> current_batch = doc_it.next();
    uint32_t local_max_doc_id = 0;
    while (!current_batch.empty()) {
      for (Document &doc : current_batch) {
        local_max_doc_id = std::max(local_max_doc_id, doc.getId());
        auto begin = doc.getData();

        tokenizer::StemmingTokenizer tokenizer(begin, doc.getSize());

        for (auto token = tokenizer.nextToken(false); !token.empty();
             token = tokenizer.nextToken(false)) {
          hyper_log_log.add(token, thread_id);
        }
      }
      current_batch = doc_it.next();
    }


    uint32_t current_max_doc_id = max_doc_id.load();
    while (current_max_doc_id < local_max_doc_id && !max_doc_id.compare_exchange_weak(current_max_doc_id, local_max_doc_id)) {
      current_max_doc_id = max_doc_id.load();
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; i++) {
    threads.emplace_back(count_distinct_tokens, i);
  }
  for (auto &thread : threads) {
    thread.join();
  }

  return {hyper_log_log.getCount(), max_doc_id};
}

std::vector<std::pair<DocumentID, double>> InvertedIndexEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  // Tokenize the query
  tokenizer::StemmingTokenizer tokenizer(query.c_str(), query.size());
  // tokenizer::SimpleTokenizer tokenizer(query.c_str(), query.size());

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
      tokens_per_document_.empty()
          ? 0.0
          : static_cast<double>(std::accumulate(
                tokens_per_document_.begin(), tokens_per_document_.end(), 0ull,
                [](uint32_t sum, const uint32_t &entry) { return sum + entry; })) /
                static_cast<double>(tokens_per_document_.size());
  return average_doc_length_;
}