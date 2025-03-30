//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include <sys/types.h>
#include <cstdint>
#include <string>
#include "fts_engine.hpp"
#include "algorithms/vector/index/hnsw/spaces/cosine_sim.hpp"
#include "index/hnsw/hnsw_alg.hpp"

namespace hnsw = vectorlib::hnsw;


class VectorEngine : public FullTextSearchEngine {
 public:
    
  VectorEngine();

  ~VectorEngine();

  void indexDocuments(std::string &data_path) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

  uint64_t footprint() override;

 private:
 	int dim = 384; 									// Dimension of the elements
	int max_elements = 10000000;				// Maximum number of elements, should be known beforehand.
	// TODO: test different parameter combinations.
	int M =  15;										// the number of bi-directional links created for every new element during construction.
	int ef_construction = 20;			// Controls index search speed/build speed tradeoff

	// TODO: Should this be a pointer?
	hnsw::HierarchicalNSW<float>* hnsw_alg;
	hnsw::CosSimSpace space;
	std::string hnsw_path = "hnsw.bin";

	uint32_t doc_count{0};
	uint64_t total_docs_length{0};
};

#endif  // VECTORSPACEMODELENGINE_HPP
