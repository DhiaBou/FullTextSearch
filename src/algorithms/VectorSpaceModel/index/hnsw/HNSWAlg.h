//
// Created by miguel on 28.11.24.
//

#ifndef FTS_HNSWALG_H
#define FTS_HNSWALG_H
//--------------------------------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <memory>
#include <random>
#include <unordered_map>

#include "../VectorSpaceLib.h"
#include "VisitedList.h"
//--------------------------------------------------------------------------------------------------
namespace vectorlib {
//--------------------------------------------------------------------------------------------------
namespace hnsw {
using tableint = unsigned int;
using linklistsizeint = unsigned int;
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
   public:
    // Constructors
    // HierarchicalNSW(SpaceInterface<dist_t> *s) {}

    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                    // bool nmslib = false,
                    size_t max_elements = 0
                    // bool allow_replace_deleted = false
    );

    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements,
                    size_t M = 16, size_t ef_construction = 200,
                    size_t random_seed = 100
                    // bool allow_replace_deleted = false
    );

    void addPoint(const void *datapoint, labeltype label,
                  bool replaceDeleted = false) override;

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void *queryData, size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr) const override;

    virtual std::vector<std::pair<dist_t, labeltype>> searchKnnCloserFirst(
        const void *queryData, size_t k,
        BaseFilterFunctor *isIdAllowed = nullptr) const;

    void saveIndex(const std::string &location);

    void setEf(size_t ef);

    ~HierarchicalNSW();

   private:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{
        0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    // mutable std::atomic<size_t> num_deleted_{0};  // number of deleted
    // elements
    // M neighbors are added as links to the new node during insertion
    size_t M_{0};
    // maximum number of links that an already existing node can
    // accumulate while other nodes are inserted and linked to it (for all
    // layers except the bottom layer)
    size_t maxM_{0};
    // maxM for layer0.
    size_t maxM0_{0};
    // during constructoin, ef nearest neighbors are discovered and become the
    // candidates to become one of the M links.
    size_t ef_construction_{0};
    size_t ef_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;
    std::vector<std::mutex> link_list_locks_;

    size_t data_size_{0};
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};

    // bool allow_replace_deleted_ =
    //     false;  // flag to replace deleted elements (marked as deleted)
    //     during
    //             // insertions

    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element
    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                   size_t max_elements_i = 0);

    void clear();

    labeltype getExternalLabel(tableint internal_id) const;
};
//--------------------------------------------------------------------------------------------------
}  // namespace hnsw
//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_HNSWALG_H
