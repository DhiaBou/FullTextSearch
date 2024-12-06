//
// Created by fts on 10/31/24.
//

#ifndef VECTORSPACEMODELENGINE_HPP
#define VECTORSPACEMODELENGINE_HPP

#include "../../fts_engine.hpp"

class VectorSpaceModelEngine : public FullTextSearchEngine {
 public:
  void indexDocuments(DocumentIterator it) override;

  std::vector<uint32_t> search(const std::string &query) override;

 private:
};

#endif  // VECTORSPACEMODELENGINE_HPP
