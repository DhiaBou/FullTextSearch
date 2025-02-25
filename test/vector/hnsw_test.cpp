#include <gtest/gtest.h>
#include <cstddef>
#include <random>
#include <vector>
#include <assert.h>

#include "algorithms/vector/index/vector_lib.hpp"
#include "algorithms/vector/index/hnsw/hnsw_alg.hpp"
#include "algorithms/vector/index/hnsw/spaces/l2_space.hpp"


using idx_t = vectorlib::labeltype;
namespace hnsw = vectorlib::hnsw;

// TODO: Add other Tests from HNSWLIB

TEST(HNSWTest, GoodRecallBeforeAndAfterSerialization) {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnsw::L2Space space(dim);
    hnsw::HierarchicalNSW<float>* alg_hnsw = new hnsw::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, idx_t>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        idx_t label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    ASSERT_GT(recall, 0.95);

    // Serialize index
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnsw::HierarchicalNSW<float>(&space, hnsw_path);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, idx_t>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        idx_t label = result.top().second;
        if (label == i) correct++;
    }
    recall = correct / max_elements;
    ASSERT_GT(recall, 0.95);

    delete[] data;
    delete alg_hnsw;
}

/// Check that we get the same result when calling searchKNN twice.
TEST(HNSWTest, DeterministicQueries) {
    int d = 4;
    idx_t n = 100;
    idx_t nq = 100;
    size_t k = 10;

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }

    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

    vectorlib::hnsw::L2Space space(d);
    vectorlib::AlgorithmInterface<float>* alg_hnsw = new vectorlib::hnsw::HierarchicalNSW<float>(&space, 2 * n);

    for (size_t i = 0; i < n; ++i) {
        alg_hnsw->addPoint(data.data() + d * i, i);
    }

    for (size_t i = 0; i < nq; ++i) {
        const void* p = query.data() + i * d;
        auto gd1 = alg_hnsw->searchKnn(p, k);
        auto gd2 = alg_hnsw->searchKnn(p, k);
        
        ASSERT_TRUE(gd1.size() == gd2.size());

        size_t t = gd1.size();
        while (!gd1.empty()) {
            ASSERT_TRUE(gd1.top() == gd2.top());
            gd1.pop();
            gd2.pop();
        }
    }

    delete alg_hnsw;
}

TEST(HNSWTest, SearchKnnCloserFirst) {
    int d = 4;
    idx_t n = 100;
    idx_t nq = 100;
    size_t k = 10;

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }

    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

    vectorlib::hnsw::L2Space space(d);
    vectorlib::AlgorithmInterface<float>* alg_hnsw = new vectorlib::hnsw::HierarchicalNSW<float>(&space, 2 * n);

    for (size_t i = 0; i < n; ++i) {
        alg_hnsw->addPoint(data.data() + d * i, i);
    }

    for (size_t i = 0; i < nq; ++i) {
        const void* p = query.data() + i * d;
        auto gd = alg_hnsw->searchKnn(p, k);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k);
        
        ASSERT_TRUE(gd.size() == res.size());

        size_t t = gd.size();
        while (!gd.empty()) {
            ASSERT_TRUE(gd.top() == res[--t]);
            gd.pop();
        }
    }

    delete alg_hnsw;
}
