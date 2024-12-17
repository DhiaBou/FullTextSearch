// Created by fts on 10/31/24.

#include "vector_engine.hpp"
#include <sys/types.h>
#include "index/hnsw/spaces/l2_space.hpp"
#include "index/hnsw/hnsw.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
//--------------------------------------------------------------------------------------------------
using namespace vectorlib::hnsw;
//--------------------------------------------------------------------------------------------------
VectorSpaceModelEngine::VectorSpaceModelEngine() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    L2Space space(dim);
    HierarchicalNSW<float>* alg_hnsw = new HierarchicalNSW<float>(&space, max_elements,
                                                                   M, ef_construction);
}
//--------------------------------------------------------------------------------------------------
void VectorSpaceModelEngine::indexDocuments(DocumentIterator it) {
    throw std::runtime_error("indexDocuments method is not yet implemented.");
}
//--------------------------------------------------------------------------------------------------
std::vector<DocumentID> VectorSpaceModelEngine::search(const std::string &query, const scoring::ScoringFunction &score_func) {
    throw std::runtime_error("search method is not yet implemented.");
}
//--------------------------------------------------------------------------------------------------
uint32_t VectorSpaceModelEngine::getDocumentCount() {
    throw std::runtime_error("Not implemented");
}
//--------------------------------------------------------------------------------------------------
double VectorSpaceModelEngine::getAvgDocumentLength() {
    throw std::runtime_error("Not implemented");
}
//--------------------------------------------------------------------------------------------------