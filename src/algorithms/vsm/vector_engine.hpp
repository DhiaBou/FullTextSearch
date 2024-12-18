//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include "../../fts_engine.hpp"
#include "./index/hnsw/hnsw.hpp"
#include "index/vector_space_lib.hpp"

namespace vectorlib {

class VectorSpaceModelEngine : public FullTextSearchEngine {
public:
//--------------------------------------------------------------------------------------------------
    VectorSpaceModelEngine(scoring::ScoringFunctionEnum sfe) : FullTextSearchEngine(sfe) {};
//--------------------------------------------------------------------------------------------------
    void indexDocuments(DocumentIterator it) override;
//--------------------------------------------------------------------------------------------------
    std::vector<DocumentID> search(const std::string &query, const scoring::ScoringFunction &score_func) override;
//--------------------------------------------------------------------------------------------------
    uint32_t getDocumentCount() override;
//--------------------------------------------------------------------------------------------------
    double getAvgDocumentLength() override;
//--------------------------------------------------------------------------------------------------

private:
    hnsw::HierarchicalNSW<float>* alg_hnsw_;
    SpaceInterface<float>* space_;
    std::vector<std::string> sortedTokens_;
    std::unordered_map<std::string, double> inverseDocumentFrequency_;
};

//--------------------------------------------------------------------------------------------------
}
#endif  // VECTORSPACEMODELENGINE_HPP
