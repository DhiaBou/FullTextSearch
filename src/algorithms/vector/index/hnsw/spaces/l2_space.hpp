//
// Created by miguel on 05.12.24.
//

#ifndef FTS_L2SPACE_H
#define FTS_L2SPACE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "algorithms/vector/index/vector_lib.hpp"

using DocumentID = uint32_t;
using TermID = uint32_t;

namespace vectorlib {
namespace hnsw {
//--------------------------------------------------------------------------------------------------
static float L2Sqr(const void* pVect1, const void* pVect2, const void* pQty) {
  float* pV1 = (float*)pVect1;
  float* pV2 = (float*)pVect2;
  size_t qty = *((size_t*)pQty);

  float res = 0;
  for (size_t i = 0; i < qty; ++i) {
    float t = *pV1 - *pV2;
    res += t * t;

    ++pV1;
    ++pV2;
  }

  return res;
}

static float L2Sqr(DocumentID docid1, DocumentID docid2,
                   std::unordered_map<DocumentID, std::vector<TermID>>& document_to_contained_terms,
                   std::unordered_map<DocumentID, std::vector<float>>& document_to_vector_,
                   const void* pVect2, const void* pQty) {
  TermID* terms1 = document_to_contained_terms[docid1].data();
  TermID* terms2 = document_to_contained_terms[docid2].data();
  float* values1 = document_to_vector_[docid1].data();
  float* values2 = document_to_vector_[docid2].data();

  float res = 0;
  while (condition) {
    // [1, 7, 19] I love you
    // [1, 7, 25] I love pizza
    if (terms1 == terms2) {
      float t = *values1 - *values2;
      res += t * t;

      ++values1;
      ++values2;
      ++terms1;
      ++terms2;
    } else if (terms1 < terms2) {
      float t = *values1;
      res += t * t;
      ++terms1;
      ++values1;
    } else {
      float t = *values2;
      res += t * t;
      ++terms2;
      ++values2;
    }
  }

  return res;
}
//--------------------------------------------------------------------------------------------------
class L2Space : public SpaceInterface<float> {
 public:
  explicit L2Space(size_t dim) {
    fstdistfunc_ = L2Sqr;
    dim_ = dim;
    data_size_ = dim * sizeof(float);
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void* get_dist_func_param() { return &dim_; }

  ~L2Space() {}

 private:
  DISTFUNC<float> fstdistfunc_;
  size_t dim_;
  size_t data_size_;
};
//--------------------------------------------------------------------------------------------------
}  // namespace hnsw
//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_L2SPACE_H
