//
// Created by fts on 10/31/24.
//

#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP

#include "../../fts_engine.hpp"

class TrigramIndexEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(std::string &data_path) override;

  std::vector<std::pair<DocumentID, double>> search(const std::string &query,
                                                    const scoring::ScoringFunction &score_func,
                                                    uint32_t num_results) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
};

#endif  // TRIGRAMINDEXENGINE_HPP
