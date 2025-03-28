// Created by fts on 10/31/24.

#include "inverted_index_engine.hpp"

#include <cmath>
#include <numeric>
#include <string>
#include <thread>

#include "datastructures/hyperloglog.hpp"
#include "documents/document_iterator.hpp"
#include "tokenizer/simpletokenizer.hpp"
#include "tokenizer/stemmingtokenizer.hpp"

void InvertedIndexEngine::indexDocuments(std::string &data_path) {
  estimateDataStructureSizes(data_path);

  DocumentIterator doc_it(data_path);

  auto index_batches = [&doc_it, this]() {
    std::vector<Document> cur_batch = doc_it.next();
    while (!cur_batch.empty()) {
      indexBatch(cur_batch);
      cur_batch = doc_it.next();
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; i++) {
    threads.emplace_back(index_batches);
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void InvertedIndexEngine::indexBatch(const std::vector<Document> &batch) {
  for (const Document &doc : batch) {
    uint32_t num_tokens = 0;
    std::unordered_map<std::string, uint32_t> local_term_frequency_per_document{};

    auto begin = doc.getData();

    tokenizer::StemmingTokenizer tokenizer(begin, doc.getSize());

    for (auto token = tokenizer.nextToken(false); !token.empty();
         token = tokenizer.nextToken(false)) {
      local_term_frequency_per_document[token]++;
      num_tokens++;
    }
    tokens_per_document_[doc.getId()] = num_tokens;

    for (const auto &[token, freq] : local_term_frequency_per_document) {
      auto add_term_frequency = [&doc, freq](std::vector<std::pair<DocumentID, uint32_t>> &docs) {
        docs.emplace_back(doc.getId(), freq);
      };
      term_frequency_per_document_.updateOrInsert(token, add_term_frequency,
                                                  std::vector<std::pair<DocumentID, uint32_t>>{});
    }
  }
}

void InvertedIndexEngine::estimateDataStructureSizes(const std::string &data_path) {
  DocumentIterator doc_it(data_path);

  std::atomic<uint32_t> max_doc_id = 0;

  HyperLogLog<std::string> hyper_log_log{NUM_THREADS};

  auto count_distinct_tokens = [&doc_it, &hyper_log_log, &max_doc_id](uint64_t thread_id) {
    std::vector<Document> current_batch = doc_it.next();
    uint32_t local_max_doc_id = 0;
    while (!current_batch.empty()) {
      for (Document &doc : current_batch) {
        local_max_doc_id = std::max(local_max_doc_id, doc.getId());
        const auto begin = doc.getData();

        tokenizer::StemmingTokenizer tokenizer(begin, doc.getSize());

        for (auto token = tokenizer.nextToken(false); !token.empty();
             token = tokenizer.nextToken(false)) {
          hyper_log_log.add(token, thread_id);
        }
      }
      current_batch = doc_it.next();
    }

    uint32_t current_max_doc_id = max_doc_id.load();
    while (current_max_doc_id < local_max_doc_id &&
           !max_doc_id.compare_exchange_weak(current_max_doc_id, local_max_doc_id)) {
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

  term_frequency_per_document_ =
      ParallelHashTable<std::string, std::vector<std::pair<DocumentID, uint32_t>>>{
          hyper_log_log.getCount() * 2};
  tokens_per_document_.resize(max_doc_id + 1);
}

std::vector<std::pair<DocumentID, double>> InvertedIndexEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  // Tokenize the query
  tokenizer::StemmingTokenizer tokenizer(query.c_str(), query.size());
  // tokenizer::SimpleTokenizer tokenizer(query.c_str(), query.size());

  // Map of doc_id -> cumulative score
  std::unordered_map<DocumentID, double> doc_to_score;

  // Compute scores for each token in the query
  for (auto token = tokenizer.nextToken(false); !token.empty();
       token = tokenizer.nextToken(false)) {
    auto it = term_frequency_per_document_.find(token);
    if (it == term_frequency_per_document_.end()) {
      // This token doesn't appear in any document
      continue;
    }

    std::vector<std::pair<DocumentID, uint32_t>> appearances = it->second;

    // For each document that contains this token, accumulate its score
    for (const auto &[doc_id, freq] : appearances) {
      double score = score_func.score({tokens_per_document_[doc_id]},
                                      {freq, static_cast<uint32_t>(appearances.size())});
      doc_to_score[doc_id] += score;
    }
  }

  // Use a min-heap to track the top 10 documents by score
  std::priority_queue<std::pair<double, uint32_t>, std::vector<std::pair<double, uint32_t>>,
                      std::greater<>>
      results;

  for (const auto &entry : doc_to_score) {
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

uint64_t InvertedIndexEngine::footprint_capacity() {
  using tokens_per_document_type = decltype(tokens_per_document_)::value_type;
  size_t tokens_per_document_footprint =
      tokens_per_document_.capacity() * sizeof(tokens_per_document_type);

  size_t token_frequency_map_footprint = term_frequency_per_document_.footprint_capacity();
  for (auto &[key, value] : term_frequency_per_document_) {
    token_frequency_map_footprint += (value.capacity() * sizeof(std::pair<uint32_t, uint32_t>));
  }
  return token_frequency_map_footprint + tokens_per_document_footprint +
         sizeof(InvertedIndexEngine);
}

uint64_t InvertedIndexEngine::footprint_size() {
  using tokens_per_document_type = decltype(tokens_per_document_)::value_type;
  size_t tokens_per_document_footprint =
      tokens_per_document_.size() * sizeof(tokens_per_document_type);

  size_t token_frequency_map_footprint = term_frequency_per_document_.footprint_size();
  for (auto &[key, value] : term_frequency_per_document_) {
    token_frequency_map_footprint += (value.size() * sizeof(std::pair<uint32_t, uint32_t>));
  }
  return token_frequency_map_footprint + tokens_per_document_footprint +
         sizeof(InvertedIndexEngine);
}

uint32_t InvertedIndexEngine::getDocumentCount() { return tokens_per_document_.size(); }

double InvertedIndexEngine::getAvgDocumentLength() {
  if (average_doc_length_ != -1) {
    return average_doc_length_;
  }
  average_doc_length_ =
      tokens_per_document_.empty()
          ? 0.0
          : static_cast<double>(
                std::accumulate(tokens_per_document_.begin(), tokens_per_document_.end(), 0ull,
                                [](uint32_t sum, const uint32_t &entry) { return sum + entry; })) /
                static_cast<double>(tokens_per_document_.size());
  return average_doc_length_;
}