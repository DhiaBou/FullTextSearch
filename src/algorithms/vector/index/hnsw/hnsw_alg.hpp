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
#include <unordered_set>

#include "../vector_lib.hpp"
#include "visited_list.hpp"
//--------------------------------------------------------------------------------------------------
namespace vectorlib {
//--------------------------------------------------------------------------------------------------
namespace hnsw {
using tableint = unsigned int;  // used for internal ids
using linklistsizeint = unsigned int;
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
   public:
    // Constructors

    // Default Constructor
    HierarchicalNSW(SpaceInterface<dist_t>* s) {};

    HierarchicalNSW(SpaceInterface<dist_t>* s, const std::string& location,
                    // bool nmslib = false,
                    size_t max_elements = 0, bool allow_replace_deleted = false);

    HierarchicalNSW(SpaceInterface<dist_t>* s, size_t max_elements, size_t M = 16,
                    size_t ef_construction = 200, size_t random_seed = 100,
                    bool allow_replace_deleted = false);

    /*
     * Adds point. Updates the point if it is already in the index.
     * If replacement of deleted elements is enabled: replaces previously deleted point if any,
     * updating it with new point
     */
    void addPoint(const void* data_point, labeltype label, bool replace_deleted = false) override;

    tableint addPoint(const void* data_point, labeltype label, int level);

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(
        const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override;

    void saveIndex(const std::string& location) override;

    void setEf(size_t ef);

    ~HierarchicalNSW();

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                                  std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

   private:
    int getRandomLevel(double reverse_size);

    void loadIndex(const std::string& location, SpaceInterface<dist_t>* s,
                   size_t max_elements_i = 0);

    void clear();

    void updatePoint(const void* dataPoint, tableint internalId, float updateNeighborProbability);

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>& top_candidates,
        const size_t M);

    void repairConnectionsForUpdate(const void* dataPoint, tableint entryPointInternalId,
                                    tableint dataPointInternalId, int dataPointLevel, int maxLevel);

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayer(tableint ep_id, const void* data_point, int layer);

    // bare_bone_search means there is no check for deletions and stop condition is ignored in
    // return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void* data_point, size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const;

    tableint mutuallyConnectNewElement(
        const void* data_point, tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>& top_candidates,
        int level, bool isUpdate);

    labeltype getExternalLabel(tableint internal_id) const;

    std::mutex& getLabelOpMutex(labeltype label) const;

    /*
     * Checks the first 16 bits of the memory to see if the element is marked deleted.
     */
    bool isMarkedDeleted(tableint internalId) const;

    /*
     * Remove the deleted mark of the node.
     */
    void unmarkDeletedInternal(tableint internalId);

    /*
     * Get direct neighbors of node with @p internalId at @p level.
     */
    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level);

    linklistsizeint* get_linklist0(tableint internal_id) const;

    linklistsizeint* get_linklist0(tableint internal_id, char* data_level0_memory_) const;

    linklistsizeint* get_linklist(tableint internal_id, int level) const;

    linklistsizeint* get_linklist_at_level(tableint internal_id, int level) const;

    void setExternalLabel(tableint internal_id, labeltype label) const;

    labeltype* getExternalLabeLp(tableint internal_id) const;

    char* getDataByInternalId(tableint internal_id) const;

    unsigned short int getListCount(linklistsizeint* ptr) const;

    void setListCount(linklistsizeint* ptr, unsigned short int size) const;

   private:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted
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

    std::mutex global;

    size_t data_size_{0};
    DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;                   // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;  // Map external labels to internal ids

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};

    // flag to replace deleted elements (marked as deleted) during insertions
    bool allow_replace_deleted_ = false;

    std::mutex deleted_elements_lock;               // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char* data_level0_memory_{nullptr};
    char** linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};
};
//--------------------------------------------------------------------------------------------------
}  // namespace hnsw
//--------------------------------------------------------------------------------------------------
}  // namespace vectorlib
//--------------------------------------------------------------------------------------------------
#endif  // FTS_HNSWALG_H
