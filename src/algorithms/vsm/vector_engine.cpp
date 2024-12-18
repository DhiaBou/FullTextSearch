// Created by fts on 10/31/24.

#include "vector_engine.hpp"

#include <sys/types.h>
#include "index/hnsw/spaces/l2_space.hpp"
#include "index/hnsw/hnsw.hpp"
#include "../../documents/document_iterator.hpp"
#include "../../tokenizer/stemmingtokenizer.hpp"
#include "../../scoring/tf_idf.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../documents/document_iterator.hpp"
#include "../../tokenizer/stemmingtokenizer.hpp"
#include "index/hnsw/hnsw.hpp"
#include "index/hnsw/spaces/l2_space.hpp"
//--------------------------------------------------------------------------------------------------
namespace vectorlib {
//--------------------------------------------------------------------------------------------------
void VectorSpaceModelEngine::indexDocuments(DocumentIterator doc_it) {
  // TODO: try other tf-idf schemes (log/double normalization) / BM25

    // set storing all tokens across all documents
    std::unordered_set<std::string> globalTokens;
    // token count per document
    std::unordered_map<DocumentID, std::unordered_map<std::string, uint32_t>> tokenCountInDocs;
    // document sizes
    std::unordered_map<DocumentID, uint32_t> docWordCounts;

    uint32_t totalWordCount_ = 0;

    // First pass: Tokenize and collect term frequencies
    while (doc_it.hasNext()) {
        auto doc = *doc_it;
        tokenizer::StemmingTokenizer tokenizer(doc->getData(), doc->getSize());
        documentCount_++;

        uint32_t docWordCount = 0;
        while (true) {
            std::string token = tokenizer.nextToken(true);
            if (token.empty()) break;

            docWordCount++;
            globalTokens.insert(token);

            // if first time seeing token in document, increment number of docs with string
            if (tokenCountInDocs[doc->getId()].count(token) == 0) {
                countDocsWithString_[token]++;
            }

            tokenCountInDocs[doc->getId()][token]++;
        }

        docWordCounts[doc->getId()] = docWordCount;
        totalWordCount_ += docWordCount;
    }

    // Sort global tokens
    sortedTokens_.assign(globalTokens.begin(), globalTokens.end());
    std::sort(sortedTokens_.begin(), sortedTokens_.end());

    initIndex(sortedTokens_.size());
    initScoringFunction();

    // Generate TF-IDF vectors and add it to index.
    for (const auto& [docId, tokenCount] : tokenCountInDocs) {
        // Allocate memory for the current document
        double* tfIdfVector = new double[sortedTokens_.size()]{0.0};

        for (size_t i = 0; i < sortedTokens_.size(); ++i) {
            const std::string& token = sortedTokens_[i];
            tfIdfVector[i] = scoringFunction_->score(
                {docWordCounts[docId]}, 
            {tokenCountInDocs[docId][token], countDocsWithString_[token]});
        }

        alg_hnsw_->addPoint(tfIdfVector, docId);

        // TODO: check if deleting vector affects index.
        delete[] tfIdfVector;
    }
}
//--------------------------------------------------------------------------------------------------
void VectorSpaceModelEngine::initIndex(int dimension) {
    // Create HNSW Index
    int M = 16;                     // Tightly connected with internal dimensionality of the data
                                    // strongly affects the memory consumption
    int ef_construction = 200;      // Controls index search speed/build speed tradeoff

    // Initing index
    space_ = new hnsw::L2Space(dimension);
    alg_hnsw_ = new hnsw::HierarchicalNSW<float>(space_, documentCount_,
                                                                   M, ef_construction);
}

//--------------------------------------------------------------------------------------------------
std::vector<DocumentID> VectorSpaceModelEngine::search(const std::string &query) {
    // Tokenize the query
    std::unordered_map<std::string, uint32_t> queryTokenCounter;
    tokenizer::StemmingTokenizer tokenizer(query.c_str(), query.size());
    uint32_t queryWordCount = 0;
    while (true) {
        std::string token = tokenizer.nextToken(false);
        if (token.empty()) break;

        queryTokenCounter[token]++;
    }


    // Generate the query TF-IDF vector
    double* queryVector = new double[sortedTokens_.size()]{0.0};
    for (size_t i = 0; i < sortedTokens_.size(); ++i) {
        const std::string& token = sortedTokens_[i];
        queryVector[i] = scoringFunction_->score({queryWordCount}, 
        {queryTokenCounter[token], countDocsWithString_[token]});
    }

  // Query
  size_t K = 10;
  auto topK = alg_hnsw_->searchKnnCloserFirst(queryVector, K);

  std::vector<DocumentID> result;
  result.reserve(K);
  for (auto vec : topK) {
    result.push_back(static_cast<DocumentID>(vec.second));
  }

  // Cleanup
  delete[] queryVector;

  return result;
}
//--------------------------------------------------------------------------------------------------
uint32_t VectorSpaceModelEngine::getDocumentCount() {
    return documentCount_;
}
//--------------------------------------------------------------------------------------------------
double VectorSpaceModelEngine::getAvgDocumentLength() {
    return static_cast<double>(totalWordCount_) / documentCount_;
}
//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib