#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
//---------------------------------------------------------------------------
#include "../../tokenizer/simpletokenizer.hpp"
#include "../../utils.hpp"
#include "models/trigram.hpp"
#include "trigram_index_engine.hpp"
//---------------------------------------------------------------------------
void TrigramIndexEngine::indexDocuments(std::string& data_path) {
  DocumentIterator doc_it(data_path);
  std::atomic<uint64_t> total_trigram_count = 0;

  auto thread_count = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  std::vector<std::unordered_map<DocumentID, uint32_t>> local_doc_to_lengths(thread_count);

  // diverge
  for (size_t i = 0; i < thread_count; ++i) {
    threads.push_back(
        std::thread([this, &doc_it, &total_trigram_count, &local_doc_to_lengths, i]() {
          total_trigram_count += consumeDocuments(doc_it, local_doc_to_lengths[i]);
        }));
  }
  for (auto& t : threads) {
    t.join();
  }

  avg_doc_length = static_cast<double>(total_trigram_count) / static_cast<double>(doc_count);

  // merge
  merge(local_doc_to_lengths);

  // compactify
  uint32_t stop_share =
      std::clamp(static_cast<uint32_t>(doc_count / (avg_doc_length + 1)), 2U, 10U);
  index.compactify(doc_count / stop_share);
}
//---------------------------------------------------------------------------
std::vector<std::pair<DocumentID, double>> TrigramIndexEngine::search(
    const std::string& query, const scoring::ScoringFunction& score_func, uint32_t num_results) {
  std::unordered_map<DocumentID, double> doc_to_score;

  const char* begin = query.c_str();
  const char* end = query.c_str() + query.size();
  trigramlib::TrigramParser trigram_parser(begin, end);

  // The query's trigrams found in the index
  std::vector<std::vector<trigramlib::DocFreq>*> trigram_results;

  // Parse the data
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
/**
 * Serialization format:
 *
 * 1. Meta-data:
 * -------------
 * 1.1 uint32_t doc_count              (4 bytes)
 * 1.2 double avg_doc_length           (8 bytes)
 * 1.3 uint32_t doc_to_length.size()   (4 bytes)
 * 1.4 For each entry in doc_to_length:
 *    a. uint32_t doc_id               (4 bytes)
 *    b. uint32_t length               (4 bytes)
 *
 * 2. Index:
 * ---------
 * 2.1 Serialized index
 */
void TrigramIndexEngine::store(const std::string& path) {
  std::ofstream file(fs::path(path), std::ios::binary);

  // store meta-data
  file.write(reinterpret_cast<const char*>(&doc_count), sizeof(doc_count));
  file.write(reinterpret_cast<const char*>(&avg_doc_length), sizeof(avg_doc_length));
  auto map_size = static_cast<uint32_t>(doc_to_length.size());
  file.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
  for (uint32_t i = 1; i < doc_to_length.size(); ++i) {
    file.write(reinterpret_cast<const char*>(&i), sizeof(i));
    auto length = doc_to_length[i];
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
  }

  // store index
  index.store(file);
}
//---------------------------------------------------------------------------
void TrigramIndexEngine::load(const std::string& path) {
  utils::FileReader file(path.c_str());
  auto it = file.begin();
  auto end = file.end();

  // load meta-data
  doc_count = *reinterpret_cast<const uint32_t*>(it);
  it += sizeof(doc_count);

  avg_doc_length = *reinterpret_cast<const double*>(it);
  it += sizeof(avg_doc_length);

  auto map_size = *reinterpret_cast<const uint32_t*>(it);
  it += sizeof(map_size);
  doc_to_length.reserve(map_size);

  for (size_t i = 0; i < map_size; ++i) {
    auto doc_id = *reinterpret_cast<const uint32_t*>(it);
    it += sizeof(doc_id);
    auto length = *reinterpret_cast<const uint32_t*>(it);
    it += sizeof(length);
    doc_to_length[doc_id] = length;
  }

  // load index
  index.load(it, end);
}
//---------------------------------------------------------------------------
uint64_t TrigramIndexEngine::footprint() {
  uint64_t size = 0;

  // Metadata
  size += sizeof(doc_count);
  size += sizeof(avg_doc_length);
  size += sizeof(doc_to_length) + doc_to_length.size() * sizeof(uint32_t);

  // Index
  size += index.footprint();

  return size;
}
//---------------------------------------------------------------------------
void TrigramIndexEngine::merge(std::vector<std::unordered_map<DocumentID, uint32_t>>& maps) {
  doc_to_length.resize(doc_count + 1);

  auto thread_count = std::thread::hardware_concurrency();
  assert(maps.size() == thread_count);

  std::vector<std::thread> threads;
  for (size_t i = 0; i < thread_count; ++i) {
    threads.push_back(std::thread([this, &maps, i]() {
      for (const auto& [doc_id, length] : maps[i]) {
        doc_to_length[doc_id] = length;
      }
    }));
  }

  for (auto& t : threads) {
    t.join();
  }
}
//---------------------------------------------------------------------------
uint64_t TrigramIndexEngine::consumeDocuments(
    DocumentIterator& doc_it, std::unordered_map<DocumentID, uint32_t>& local_doc_to_length) {
  uint64_t local_trigram_count = 0;
  uint32_t local_doc_count = 0;

  std::vector<Document> docs = doc_it.next();
  while (!docs.empty()) {
    for (const auto& doc : docs) {
      uint32_t doc_length = 0;
      // Count the number of occurences per trigram for the scoring.
      // Note: Since the hash and equality function of the trigram do NOT
      // differentiate word offset, which is wanted here, the trigram's
      // raw value is used.
      std::unordered_map<uint32_t, uint32_t> trigram_occurences;

      const char* begin = doc.getData();
      const char* end = doc.getData() + doc.getSize();
      trigramlib::TrigramParser trigram_parser(begin, end);

      // Parse the data
      while (trigram_parser.hasNext()) {
        ++trigram_occurences[trigram_parser.next().getRawValue()];
        ++doc_length;
      }

      // Insert into index
      for (const auto& [raw_trigram, count] : trigram_occurences) {
        index.insert(trigramlib::Trigram(raw_trigram), {doc.getId(), count});
      }

      // Update statistics
      local_trigram_count += doc_length;
      local_doc_to_length[doc.getId()] = doc_length;
      ++local_doc_count;
    }
    docs = doc_it.next();
  }
  doc_count += local_doc_count;

  return local_trigram_count;
}
//---------------------------------------------------------------------------
uint32_t TrigramIndexEngine::getDocumentCount() { return doc_count; }
//---------------------------------------------------------------------------
double TrigramIndexEngine::getAvgDocumentLength() { return avg_doc_length; }