//
// Created by fts on 10/31/24.
//

#ifndef INVERTEDINDEXENGINE_HPP
#define INVERTEDINDEXENGINE_HPP

#include "../../fts_engine.hpp"

class InvertedIndexEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(DocumentIterator it) override;

  std::vector<DocumentID> search(const std::string &query,
                                 const scoring::ScoringFunction &score_func) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
};

#endif  // INVERTEDINDEXENGINE_HPP
