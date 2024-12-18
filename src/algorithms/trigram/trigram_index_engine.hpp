//
// Created by fts on 10/31/24.
//

#ifndef TRIGRAMINDEXENGINE_HPP
#define TRIGRAMINDEXENGINE_HPP

#include "../../fts_engine.hpp"

class TrigramIndexEngine : public FullTextSearchEngine {
 public:
  TrigramIndexEngine(scoring::ScoringFunctionEnum sfe): FullTextSearchEngine(sfe) {};

  void indexDocuments(DocumentIterator it) override;

  std::vector<DocumentID> search(const std::string &query,
                                 const scoring::ScoringFunction &score_func) override;

  uint32_t getDocumentCount() override;

  double getAvgDocumentLength() override;

 private:
};

#endif  // TRIGRAMINDEXENGINE_HPP
