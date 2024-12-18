// Created by fts on 10/31/24.

#include "vector_engine.hpp"
#include <sys/types.h>
#include "index/hnsw/spaces/l2_space.hpp"
#include "index/hnsw/hnsw.hpp"
#include "../../documents/document_iterator.hpp"
#include "../../tokenizer/stemmingtokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
//--------------------------------------------------------------------------------------------------
namespace vectorlib {
//--------------------------------------------------------------------------------------------------
void VectorSpaceModelEngine::indexDocuments(DocumentIterator doc_it) {
    // TODO: try other tf-idf schemes (log/double normalization) / BM25

    std::unordered_set<std::string> globalTokens;
    std::unordered_map<DocumentID, std::unordered_set<std::string>> tokensPerDocument;
    std::unordered_map<DocumentID, std::unordered_map<std::string, double>> termFrequencies; // TF storage
    std::unordered_map<std::string, DocumentID> documentFrequency; // DF storage
    uint32_t totalDocuments = 0;

    // First pass: Tokenize and collect term frequencies
    while (doc_it.hasNext()) {
        auto doc = *doc_it;
        const char* begin = doc->getData();
        uint32_t doc_id = doc->getId();
        tokenizer::StemmingTokenizer tokenizer(begin, doc->getSize());
        totalDocuments++;

        std::unordered_map<std::string, DocumentID> termCounts;
        uint32_t docTokenCount = 0;

        while (true) {
            std::string token = tokenizer.nextToken(true);
            if (token.empty()) break;

            globalTokens.insert(token);
            tokensPerDocument[doc_id].insert(token);
            termCounts[token]++;
            docTokenCount++;
        }

        // Calculate TF for the document
        for (auto& [term, count] : termCounts) {
            termFrequencies[doc_id][term] = static_cast<double>(count) / docTokenCount;
        }

        // Update document frequency (DF) for each term
        for (const auto& term : tokensPerDocument[doc_id]) {
            documentFrequency[term]++;
        }
    }

    // Sort global tokens
    sortedTokens_.assign(globalTokens.begin(), globalTokens.end());
    std::sort(sortedTokens_.begin(), sortedTokens_.end());
    size_t vocabularySize = sortedTokens_.size();

    // Map to hold TF-IDF vectors for each document
    std::unordered_map<uint32_t, double*> tfIdfVectors;

    // Allocate memory and calculate IDF
    for (const auto& term : globalTokens) {
        inverseDocumentFrequency_[term] = std::log(static_cast<double>(totalDocuments) / documentFrequency[term]);
    }

    // Generate TF-IDF vectors
    for (const auto& [doc_id, tokens] : tokensPerDocument) {
        // Allocate memory for the current document
        double* tfIdfVector = new double[vocabularySize]{0.0};

        for (size_t i = 0; i < vocabularySize; ++i) {
            const std::string& term = sortedTokens_[i];
            double tf = termFrequencies[doc_id].count(term) ? termFrequencies[doc_id][term] : 0.0;
            double idf = inverseDocumentFrequency_[term];
            tfIdfVector[i] = tf * idf;
        }

        tfIdfVectors[doc_id] = tfIdfVector; // Associate the document ID with its TF-IDF vector
    }

    // Create HNSW Index
    int dim = vocabularySize;   // Dimension of the elements
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    space_ = new  hnsw::L2Space(dim);
    alg_hnsw_ = new hnsw::HierarchicalNSW<float>(space_, totalDocuments,
                                                                   M, ef_construction);

    // Add data to index
    for (auto [doc_id, data]: tfIdfVectors) {
        alg_hnsw_->addPoint(data, doc_id);
    }

    // Cleanup allocated memory
    for (auto [doc_id, data]: tfIdfVectors) {
        delete[] tfIdfVectors[doc_id];
    }
}
//--------------------------------------------------------------------------------------------------
std::vector<DocumentID> VectorSpaceModelEngine::search(const std::string &query, const scoring::ScoringFunction &) {
    // Tokenize the query
    std::unordered_map<std::string, uint32_t> queryTermCounts;
    uint32_t queryTokenCount = 0;
    tokenizer::StemmingTokenizer tokenizer(query.c_str(), query.size());

    while (true) {
        std::string token = tokenizer.nextToken(false);
        if (token.empty()) break;

        if (queryTermCounts.find(token) == queryTermCounts.end()) {
            queryTermCounts[token] = 0;
        }
        queryTermCounts[token]++;
        queryTokenCount++;
    }

    // Calculate TF for the query
    std::unordered_map<std::string, double> queryTF;
    for (const auto& [term, count] : queryTermCounts) {
        queryTF[term] = static_cast<double>(count) / queryTokenCount;
    }

    // Generate the query TF-IDF vector
    double* queryVector = new double[sortedTokens_.size()]{0.0};
    for (size_t i = 0; i < sortedTokens_.size(); ++i) {
        const std::string& term = sortedTokens_[i];
        double tf = queryTF.count(term) ? queryTF[term] : 0.0;
        double idf = inverseDocumentFrequency_.count(term) ? inverseDocumentFrequency_[term] : 0.0;
        queryVector[i] = tf * idf;
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
    throw std::runtime_error("Not implemented");
}
//--------------------------------------------------------------------------------------------------
double VectorSpaceModelEngine::getAvgDocumentLength() {
    throw std::runtime_error("Not implemented");
}
//--------------------------------------------------------------------------------------------------
}