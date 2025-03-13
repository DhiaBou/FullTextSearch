#ifndef FTS_COSINE_SIM_H
#define FTS_COSINE_SIM_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include "algorithms/vector/index/vector_lib.hpp"
//--------------------------------------------------------------------------------------------------
namespace vectorlib {
//--------------------------------------------------------------------------------------------------
namespace hnsw {
//--------------------------------------------------------------------------------------------------
static float CosSimilarity(const void* pVect1, const void* pVect2, const void* pQty) {
    const float* pV1 = static_cast<const float*>(pVect1);
    const float* pV2 = static_cast<const float*>(pVect2);
    size_t qty = *static_cast<const size_t*>(pQty);

    float dotProd = 0;
    float lengthV1 = 0;
    float lengthV2 = 0;
    for (size_t i = 0; i < qty; ++i) {
        float el1 = *pV1;
        float el2 = *pV2;
        dotProd += el1 * el2;
        lengthV1 += el1 * el1;
        lengthV2 += el2 * el2;

        ++pV1;
        ++pV2;
    }

    // Avoid division by zero
    assert(lengthV1 != 0 && lengthV2 != 0);

    // negate because we want to minimize the distance (maximize similarity)
    return 1 - dotProd / (std::sqrt(lengthV1) * sqrt(lengthV2));
}
//--------------------------------------------------------------------------------------------------
class CosSimSpace : public SpaceInterface<float> {
public:
    explicit CosSimSpace(size_t dim) {
        fstdistfunc_ = CosSimilarity;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() { return data_size_; }

    DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

    void* get_dist_func_param() { return &dim_; }

    ~CosSimSpace() {}

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
#endif // FTS_COSINE_SIM_H