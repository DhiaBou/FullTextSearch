#ifndef FTS_L2SPACEDOCID_H
#define FTS_L2SPACEDOCID_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "algorithms/vector/index/vector_lib.hpp"

using DocumentID = uint32_t;
using TermID = uint32_t;

namespace vectorlib {
namespace hnsw {

//--------------------------------------------------------------------------------------------------

class L2DistanceFunctor {
 public:
  // Constructor takes references to external objects
  L2DistanceFunctor(const std::vector<std::vector<float>>& document_to_vector,
                    const std::vector<std::vector<TermID>>& document_to_contained_terms)
      : document_to_vector_(document_to_vector),
        document_to_contained_terms_(document_to_contained_terms) {}

  // Overload operator() to make this object callable like a function
  float operator()(const void* pVect1, const void* pVect2, const void* pQty) const {
    DocumentID docid1 = *((DocumentID*)pVect1);
    DocumentID docid2 = *((DocumentID*)pVect2);

    const auto& terms1 = document_to_contained_terms_.at(docid1);
    const auto& terms2 = document_to_contained_terms_.at(docid2);
    const auto& values1 = document_to_vector_.at(docid1);
    const auto& values2 = document_to_vector_.at(docid2);

    float res = 0;

    auto terms1_it = terms1.begin();
    auto terms2_it = terms2.begin();
    auto values1_it = values1.begin();
    auto values2_it = values2.begin();

    while (terms1_it != terms1.end() && terms2_it != terms2.end()) {
      if (*terms1_it == *terms2_it) {
        float t = *values1_it - *values2_it;
        res += t * t;
        ++values1_it;
        ++values2_it;
        ++terms1_it;
        ++terms2_it;
      } else if (*terms1_it < *terms2_it) {
        float t = *values1_it;
        res += t * t;
        ++terms1_it;
        ++values1_it;
      } else {
        float t = *values2_it;
        res += t * t;
        ++terms2_it;
        ++values2_it;
      }
    }

    return res;
  }

 private:
  const std::vector<std::vector<float>>& document_to_vector_;
  const std::vector<std::vector<TermID>>& document_to_contained_terms_;
};

// static float l2SqrDocId(const void* pVect1, const void* pVect2, const void* pQty) {
//   // DocumentID docid1, DocumentID docid2,
//   // std::unordered_map<DocumentID, std::vector<TermID>>& document_to_contained_terms,
//   // std::unordered_map<DocumentID, std::vector<float>>& document_to_vector_, const void* pVect2,
//   // const void* pQty) {

//   DocumentID docid1 = *((DocumentID*)pVect1);
//   DocumentID docid2 = *((DocumentID*)pVect2);

//   TermID* terms1 = document_to_contained_terms_[docid1].data();
//   TermID* terms2 = document_to_contained_terms_[docid2].data();
//   float* values1 = document_to_vector_[docid1].data();
//   float* values2 = document_to_vector_[docid2].data();

//   float res = 0;

//   // TODO: Change 'false' to something smart
//   // while (false) {
//   // [1, 7, 19] I love you
//   // [1, 7, 25] I love pizza
//   if (terms1 == terms2) {
//     float t = *values1 - *values2;
//     res += t * t;

//     ++values1;
//     ++values2;
//     ++terms1;
//     ++terms2;
//   } else if (terms1 < terms2) {
//     float t = *values1;
//     res += t * t;
//     ++terms1;
//     ++values1;
//   } else {
//     float t = *values2;
//     res += t * t;
//     ++terms2;
//     ++values2;
//   }
//   // }

//   return res;
// }
// };  // namespace hnsw
//--------------------------------------------------------------------------------------------------
class L2SpaceDocId : public SpaceInterface<float> {
 public:
  explicit L2SpaceDocId(size_t dim, std::vector<std::vector<float>>& document_to_vector_,
                        std::vector<std::vector<TermID>>& document_to_contained_terms_)
      : document_to_vector_(document_to_vector_),
        document_to_contained_terms_(document_to_contained_terms_),
        l2_distance_functor_(document_to_vector_, document_to_contained_terms_) {
    fstdistfunc_ = std::ref(l2_distance_functor_);
    dim_ = dim;
    data_size_ = dim * sizeof(float);
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void* get_dist_func_param() { return &dim_; }

  ~L2SpaceDocId() {}

  std::vector<std::vector<float>>& document_to_vector_;
  std::vector<std::vector<TermID>>& document_to_contained_terms_;

 private:
  // DISTFUNC<float> fstdistfunc_;
  size_t dim_;
  size_t data_size_;

  L2DistanceFunctor l2_distance_functor_;
  std::function<float(const void*, const void*, const void*)> fstdistfunc_;
};

//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
}  // namespace hnsw
//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_L2SPACE_H
