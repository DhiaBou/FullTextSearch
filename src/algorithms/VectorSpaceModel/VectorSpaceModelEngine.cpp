// Created by fts on 10/31/24.

#include "VectorSpaceModelEngine.hpp"
#include "./index/VectorSpaceLib.h"
#include "index/hnsw/spaces/L2Space.h"
#include "index/hnsw/HNSWAlg.h"

#include <stdexcept>
#include <string>

using namespace vectorlib::hnsw;

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


void VectorSpaceModelEngine::indexDocuments(DocumentIterator it) {
    throw std::runtime_error("indexDocuments method is not yet implemented.");
}

std::vector<std::shared_ptr<Document> > VectorSpaceModelEngine::search(const std::string &query) {
    throw std::runtime_error("search method is not yet implemented.");
}
