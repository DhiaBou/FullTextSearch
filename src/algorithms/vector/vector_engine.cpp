#include "vector_engine.hpp"
#include "./index/hnsw/hnsw_alg.hpp"
#include "algorithms/vector/index/hnsw/spaces/l2_space.hpp"

#include <stdexcept>
#include <string>
//---------------------------------------------------------------------------
VectorEngine::VectorEngine(): space(dim) {
  model.loadModel(model_path);
}
//---------------------------------------------------------------------------
void VectorEngine::indexDocuments(DocumentIterator it) {
  throw std::runtime_error("indexDocuments method is not yet implemented.");
}
//---------------------------------------------------------------------------
std::vector<std::pair<DocumentID, double>> VectorEngine::search(
    const std::string &query, const scoring::ScoringFunction &score_func, uint32_t num_results) {
  throw std::runtime_error("search method is not yet implemented.");
}
//---------------------------------------------------------------------------
uint32_t VectorEngine::getDocumentCount() {
  throw std::runtime_error("Method is not yet implemented.");
}
//---------------------------------------------------------------------------
double VectorEngine::getAvgDocumentLength() {
  throw std::runtime_error("Method is not yet implemented.");
}
