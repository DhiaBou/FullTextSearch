//
// Created by miguel on 28.11.24.
//

#ifndef FTS_VECTORSPACELIB_H
#define FTS_VECTORSPACELIB_H

#include <cstddef>
#include <queue>
#include <string>
#include <algorithm>
#include <iostream>

namespace vectorlib {
//--------------------------------------------------------------------------------------------------
using labeltype = size_t;
//--------------------------------------------------------------------------------------------------
template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}
//--------------------------------------------------------------------------------------------------
// This can be extended to store state for filtering (e.g. from a std::set)
class BaseFilterFunctor {
   public:
    virtual bool operator()(vectorlib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() = default;
};
//--------------------------------------------------------------------------------------------------
template <typename MTYPE>
using DISTFUNC = MTYPE(*)(const void*, const void*, const void*);
//--------------------------------------------------------------------------------------------------
template<typename MTYPE>
class SpaceInterface {
   public:
    virtual size_t get_data_size() = 0;

    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface() = default;
};
//--------------------------------------------------------------------------------------------------
template<typename dist_t>
class AlgorithmInterface {
   public:
    virtual void addPoint(const void *datapoint, labeltype label, bool replaceDeleted = false) = 0;

    virtual std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer first
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    virtual void saveIndex(const std::string &location) = 0;

    virtual ~AlgorithmInterface() = default;
};
//--------------------------------------------------------------------------------------------------
} // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_VECTORSPACELIB_H
