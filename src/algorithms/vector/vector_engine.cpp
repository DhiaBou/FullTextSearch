#include "vector_engine.hpp"
#include "./index/hnsw/hnsw_alg.hpp"
#include "algorithms/vector/index/hnsw/spaces/l2_space.hpp"
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"
#include "fts_engine.hpp"
#include "vector.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <thread>

//---------------------------------------------------------------------------
using idx_t = vectorlib::labeltype;
namespace hnsw = vectorlib::hnsw;
//---------------------------------------------------------------------------
VectorEngine::VectorEngine(): space(dim) {
  hnsw_alg = new hnsw::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
  std::cout << "Loading Model" << std::endl;
  model.loadModel(model_path);
  std::cout << "Loaded Model" << std::endl;
}
//---------------------------------------------------------------------------
VectorEngine::~VectorEngine() {
  delete hnsw_alg;
}
//---------------------------------------------------------------------------
static void getEmbedding(fasttext::FastText& model, const char* text, fasttext::Vector& fasttext_embedding) {
    std::istringstream text_stream(text);
    model.getSentenceVector(text_stream, fasttext_embedding);
}
//---------------------------------------------------------------------------
void VectorEngine::indexDocuments(std::string &data_path) {
  auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  DocumentIterator doc_it(data_path);
  size_t NUM_THREADS = 8;

  int counter = 0;
  
  std::vector<std::thread> workers;
  workers.reserve(NUM_THREADS);
  for (size_t i = 0; i < NUM_THREADS; ++i) {
    workers.emplace_back([&]() {
      fasttext::Vector embedding(model.getDimension());  // Local embedding per thread
      std::vector<Document> current_batch = doc_it.next();

      while (!current_batch.empty()) {

        for (Document &doc : current_batch) {
          std::cout << std::format("#{} - DocID {} of size {}", doc_count,  doc.getId(), doc.getSize()) << std::endl;
          
          getEmbedding(model, doc.getData(), embedding);

          hnsw_alg->addPoint(embedding.data(), doc.getId());

          ++doc_count;
          total_docs_length += doc.getSize();
        }

        current_batch = doc_it.next();
        counter += 128;

        std::cout << std::format("Indexed around {} documents so far.\n", counter);

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
    
    fasttext::Vector embedding(model.getDimension());
    getEmbedding(model, query.c_str(), embedding);
    
    std::priority_queue<std::pair<float, idx_t>> pq = hnsw_alg->searchKnn(embedding.data(), num_results);

    std::vector<std::pair<DocumentID, double>> res;
    res.resize(num_results);
    size_t i = num_results;
    while (!pq.empty()) {
      auto [dist, id] = pq.top();

      res[--i] = {id, dist};
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
