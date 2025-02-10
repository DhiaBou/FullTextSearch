//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include "../../fts_engine.hpp"

class VectorSpaceModelEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(DocumentIterator it) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  void store(const std::string &path) override;

  void load(const std::string &path) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
};

#endif  // VECTORSPACEMODELENGINE_HPP
