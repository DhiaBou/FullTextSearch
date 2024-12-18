//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include <cstdint>
#include <memory>
#include "../../fts_engine.hpp"
#include "./index/hnsw/hnsw.hpp"
#include "index/vector_space_lib.hpp"

namespace vectorlib {

class VectorSpaceModelEngine : public FullTextSearchEngine {
public:
//--------------------------------------------------------------------------------------------------
    VectorSpaceModelEngine(scoring::ScoringFunctionEnum sfe) : FullTextSearchEngine(sfe), documentCount_{0}, totalWordCount_{0} {};
//--------------------------------------------------------------------------------------------------
    void indexDocuments(DocumentIterator it) override;
//--------------------------------------------------------------------------------------------------
    std::vector<DocumentID> search(const std::string &query) override;
//--------------------------------------------------------------------------------------------------
    uint32_t getDocumentCount() override;
//--------------------------------------------------------------------------------------------------
    double getAvgDocumentLength() override;
//--------------------------------------------------------------------------------------------------
private:
//--------------------------------------------------------------------------------------------------
    void initIndex(int dimension);
//--------------------------------------------------------------------------------------------------
private:
    hnsw::HierarchicalNSW<float>* alg_hnsw_;
    SpaceInterface<float>* space_;
    std::vector<std::string> sortedTokens_;
    // number of documents the string occurs.
    std::unordered_map<std::string, uint32_t> countDocsWithString_;
    uint32_t documentCount_;
    uint32_t totalWordCount_;
    double avgDocLength_;
    std::unique_ptr<scoring::ScoringFunction> scoringFunction_;
};

//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib
#endif  // VECTORSPACEMODELENGINE_HPP
