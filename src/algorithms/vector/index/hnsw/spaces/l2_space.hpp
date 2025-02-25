//
// Created by miguel on 05.12.24.
//

#ifndef FTS_L2SPACE_H
#define FTS_L2SPACE_H
#include <cstddef>
#include "algorithms/vector/index/vector_lib.hpp"

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
} // namespace hnsw
//--------------------------------------------------------------------------------------------------
} // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_L2SPACE_H

