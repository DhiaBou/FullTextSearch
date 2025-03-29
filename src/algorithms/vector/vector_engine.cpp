#include "vector_engine.hpp"
#include "./index/hnsw/hnsw_alg.hpp"
#include "algorithms/vector/index/hnsw/spaces/l2_space.hpp"
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"
#include "fts_engine.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <curl/curl.h>
#include <curl/easy.h>
#include <json/json.h>
#include <json/writer.h>
#include <format>
//---------------------------------------------------------------------------
using idx_t = vectorlib::labeltype;
namespace hnsw = vectorlib::hnsw;
//---------------------------------------------------------------------------
VectorEngine::VectorEngine(): space(dim) {
  hnsw_alg = new hnsw::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
  // std::cout << "Vector Engine Initialized" << std::endl;
}
//---------------------------------------------------------------------------
VectorEngine::~VectorEngine() {
  hnsw_alg->saveIndex("hnswIndex.bin");
  delete hnsw_alg;
}
//---------------------------------------------------------------------------
// Function to combine document ID and chunk ID into a single index ID
inline idx_t combineIds(uint32_t docId, uint32_t chunkId) {
    // Use the lower 32 bits for the document ID and the next 32 bits for the chunk ID
    return (static_cast<idx_t>(chunkId) << 32) | static_cast<idx_t>(docId);
}
//---------------------------------------------------------------------------
// Function to extract document ID from a combined ID
inline uint32_t getDocId(idx_t combinedId) {
    return static_cast<uint32_t>(combinedId & 0xFFFFFFFF);
}
//---------------------------------------------------------------------------
// Function to extract chunk ID from a combined ID
inline uint32_t getChunkId(idx_t combinedId) {
    return static_cast<uint32_t>(combinedId >> 32);
}
//---------------------------------------------------------------------------
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
  size_t total_size = size * nmemb;
  output->append((char*)contents, total_size);
  return total_size;
}
//---------------------------------------------------------------------------
/// Function to get embedding for a single chunk
static bool getSingleEmbedding(const std::string& text, std::vector<float>& embedding) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Failed to initialize cURL\n";
    return false;
  }
  
  Json::Value json_payload;
  json_payload["content"] = text;
  
  Json::StreamWriterBuilder writer;
  std::string json_data = Json::writeString(writer, json_payload);
  
  std::string response_string;
  
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  
  curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8080/embedding");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
  
  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << "cURL request failed: " << curl_easy_strerror(res) << std::endl;
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return false;
  }
  
  // Check HTTP response code
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  if (http_code != 200) {
    std::cerr << "HTTP error: " << http_code << std::endl;
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return false;
  }
  
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  
  // Skip parsing if response is empty
  if (response_string.empty()) {
    std::cerr << "Empty response received from the API" << std::endl;
    return false;
  }
  
  // Check if the response is valid JSON before parsing
  if (response_string[0] != '{' && response_string[0] != '[') {
    std::cerr << "Invalid JSON response: " << response_string << std::endl;
    return false;
  }
  
  // Parse JSON response
  Json::CharReaderBuilder reader;
  Json::Value root;
  std::string errs;
  
  std::istringstream s(response_string);
  if (!Json::parseFromStream(reader, s, &root, &errs)) {
    std::cerr << "JSON Parsing Error: " << errs << std::endl;
    return false;
  }
  
  if (root.isArray() && !root.empty() && root[0].isMember("embedding")) {
    const Json::Value& emb_vec = root[0]["embedding"];
    embedding.clear();
    for (const auto& val : emb_vec[0]) {
      embedding.push_back(val.asFloat());
    }
    return true;
  } else {
    std::cerr << "Unexpected JSON format\n";
    return false;
  }
}
//---------------------------------------------------------------------------
static bool getEmbeddings(const char* text, size_t text_length,
                          std::vector<std::vector<float>>& chunk_embeddings) {
    const size_t MAX_CHUNK_SIZE = 300; // characters
    
    // If text is small enough, process as a single chunk
    if (text_length <= MAX_CHUNK_SIZE) {
        std::vector<float> embedding;
        if (getSingleEmbedding(std::string(text, text_length), embedding)) {
            chunk_embeddings.push_back({embedding});
            return true;
        }
        return false;
    }
        
    // Text is to long, split into manageable chunks
    for (size_t i = 0; i < text_length; i += MAX_CHUNK_SIZE) {
      size_t chunk_size = std::min(MAX_CHUNK_SIZE, text_length - i);
      
      // Skip empty chunks
      if (chunk_size == 0) {
          continue;
      }
      
      // Create a temporary null-terminated string for this chunk
      std::string chunk(text + i, chunk_size);
      
      std::vector<float> chunk_embedding;
      if (getSingleEmbedding(chunk, chunk_embedding)) {
          chunk_embeddings.push_back({chunk_embedding});
      } else {
          std::cerr << "Failed to get embedding for a chunk, skipping" << std::endl;
      }
    }
    
    // If we couldn't get any embeddings, fail
    if (chunk_embeddings.empty()) {
      std::cerr << "Failed to get any chunk embeddings" << std::endl;
      return false;
    }
    
    return true;
}
//---------------------------------------------------------------------------
void VectorEngine::indexDocuments(std::string &data_path) {
  auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  DocumentIterator doc_it(data_path);
  size_t NUM_THREADS = 8;
  
  std::vector<std::thread> workers;
  workers.reserve(NUM_THREADS);
  for (size_t i = 0; i < NUM_THREADS; ++i) {
    workers.emplace_back([&]() {
      std::vector<std::vector<float>> chunk_embeddings;

      std::vector<Document> current_batch = doc_it.next();

      while (!current_batch.empty()) {

        for (Document &doc : current_batch) {
          DocumentID doc_id = doc.getId();
          ++doc_count;

          chunk_embeddings.clear();

          if (getEmbeddings(doc.getData(), doc.getSize(),  chunk_embeddings)) {
            // Add each chunk to the index
            for (size_t chunk_id = 0; chunk_id < chunk_embeddings.size(); chunk_id++) {
              hnsw_alg->addPoint(chunk_embeddings[chunk_id].data(), combineIds(doc_id, chunk_id));
            }
            total_docs_length += doc.getSize();
          } else {
            std::cerr << "Failed to get embeddings for doc: " << doc.getId() << std::endl;
          }
        }

        current_batch = doc_it.next();
      }
    });
  }

  for (auto &worker: workers) {
    worker.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now(); // End timing
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  std::cout << std::format("Indexing completed in {:.2f} seconds\n", elapsed_time.count());
}
//---------------------------------------------------------------------------
std::vector<std::pair<DocumentID, double>> VectorEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
    
      std::vector<std::vector<float>> chunk_embeddings;
    if (!getEmbeddings(query.c_str(), query.length(), chunk_embeddings)) {
      std::cerr << "Failed to get embedding for search query.\n";
      return {};
    }
    
    std::priority_queue<std::pair<float, idx_t>> pq = hnsw_alg->searchKnn(chunk_embeddings[0].data(), num_results);

    std::vector<std::pair<DocumentID, double>> res;
    res.resize(num_results);
    size_t i = num_results;
    while (!pq.empty()) {
      auto [dist, id] = pq.top();

      res[--i] = {getDocId(id), dist};
      pq.pop();
    }

    return res;
}
//---------------------------------------------------------------------------
uint32_t VectorEngine::getDocumentCount() {
  return doc_count;
}
//---------------------------------------------------------------------------
double VectorEngine::getAvgDocumentLength() {
  return static_cast<double>(total_docs_length) / static_cast<double>(doc_count);
}
//---------------------------------------------------------------------------
uint64_t VectorEngine::footprint() {
  return 0;
}
