//
// Created by miguel on 28.11.24.
//

#ifndef FTS_VECTORSPACELIB_H
#define FTS_VECTORSPACELIB_H

#include <cstddef>
#include <iostream>
#include <queue>
#include <string>

namespace vectorlib {
//--------------------------------------------------------------------------------------------------
// TODO: Check if we really need external labels and internal ids? Why not one for both!?
using labeltype = size_t;
//--------------------------------------------------------------------------------------------------
template <typename T>
static void readBinaryPOD(std::istream& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}
//--------------------------------------------------------------------------------------------------
template <typename T>
static void writeBinaryPOD(std::ostream& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
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
using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);
//--------------------------------------------------------------------------------------------------
template <typename MTYPE>
class SpaceInterface {
   public:
    virtual size_t get_data_size() = 0;

    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    virtual void* get_dist_func_param() = 0;

    virtual ~SpaceInterface() = default;
};
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
class BaseSearchStopCondition {
   public:
    virtual void add_point_to_result(labeltype label, const void* datapoint, dist_t dist) = 0;

    virtual void remove_point_from_result(labeltype label, const void* datapoint, dist_t dist) = 0;

    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_remove_extra() = 0;

    virtual void filter_results(std::vector<std::pair<dist_t, labeltype>>& candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
class AlgorithmInterface {
   public:
    virtual void addPoint(const void* datapoint, labeltype label, bool replaceDeleted = false) = 0;

    virtual std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer first
    // This function is not a pure virtual function (does not have to be override implementation)
    virtual std::vector<std::pair<dist_t, labeltype>> searchKnnCloserFirst(
        const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    virtual void saveIndex(const std::string& location) = 0;

    virtual ~AlgorithmInterface() = default;
};
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>> AlgorithmInterface<dist_t>::searchKnnCloserFirst(
    const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_VECTORSPACELIB_H
