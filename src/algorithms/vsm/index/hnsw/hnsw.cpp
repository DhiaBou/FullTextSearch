//
// Created by miguel on 05.12.24.
//

#include "hnsw.hpp"
#include "../vector_space_lib.hpp"
#include "visited_list.hpp"

#include <assert.h>
#include <fstream>


// TODO: Have same order of functions in header and source file
namespace vectorlib {
namespace hnsw {
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t>* s, const std::string& location,
                                         // bool nmslib,
                                         size_t max_elements, bool allow_replace_deleted)
    : allow_replace_deleted_(allow_replace_deleted) {
    loadIndex(location, s, max_elements);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t>* s, size_t max_elements, size_t M,
                                         size_t ef_construction, size_t random_seed,
                                         bool allow_replace_deleted)
    : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
      link_list_locks_(max_elements),
      element_levels_(max_elements),
      allow_replace_deleted_(allow_replace_deleted) {
    max_elements_ = max_elements;
    // num_deleted_ = 0;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
        M_ = M;
    } else {
        std::cout << "warning: M parameter exceeds 10000 which may lead to "
                     "adverse effects.\n";
        std::cout << "Cap to 10000 will be applied for the rest of the processing.\n";
        M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    // consider at least M candidates.
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;  // TODO: why 10?

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    // TODO: replace this with structs?
    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);

    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");

    cur_element_count = 0;

    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
}

//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::~HierarchicalNSW() {
    clear();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::mutex& HierarchicalNSW<dist_t>::getLabelOpMutex(labeltype label) const {
    // calculate hash
    size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
    return label_op_locks_[lock_id];
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
char* HierarchicalNSW<dist_t>::getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
labeltype* HierarchicalNSW<dist_t>::getExternalLabeLp(tableint internal_id) const {
    return (labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
           sizeof(labeltype));
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::setListCount(linklistsizeint* ptr, unsigned short size) const {
    *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
unsigned short int HierarchicalNSW<dist_t>::getListCount(linklistsizeint* ptr) const {
    return *((unsigned short int*)ptr);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::vector<tableint> HierarchicalNSW<dist_t>::getConnectionsWithLock(tableint internalId,
                                                                      int level) {
    std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
    unsigned int* data = get_linklist_at_level(internalId, level);
    int size = getListCount(data);
    std::vector<tableint> result(size);
    tableint* ll = (tableint*)(data + 1);
    memcpy(result.data(), ll, size * sizeof(tableint));
    return result;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidateSet;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidateSet.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        // TODO: use this instead of the monstruosity below
        // int *data = (int*)get_linklist_at_level(curNodeNum, layer);  // = (int *)(linkList0_ +
        // curNodeNum * size_links_per_element0_);
        int* data;
        if (layer == 0) {
            data = (int*)get_linklist0(curNodeNum);
        } else {
            data = (int*)get_linklist(curNodeNum, layer);
            //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) *
            //                    size_links_per_element_);
        }
        size_t size = getListCount((linklistsizeint*)data);
        tableint* datal = (tableint*)(data + 1);

        for (int j = 0; j < size; ++j) {
            tableint candidate_id = *(datal + j);
            //                    if (candidate_id == 0) continue;

            if (visited_array[candidate_id] == visited_array_tag) continue;
            visited_array[candidate_id] = visited_array_tag;
            char* currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
            if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);

                if (!isMarkedDeleted(candidate_id)) top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef_construction_) top_candidates.pop();

                if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
tableint HierarchicalNSW<dist_t>::mutuallyConnectNewElement(
    const void* data_point, tableint cur_c,
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>& top_candidates,
    int level, bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selectedNeighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
        // lock only during the update
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
        if (isUpdate) {
            lock.lock();
        }
        linklistsizeint* ll_cur;
        // TODO: replace with better function
        if (level == 0)
            ll_cur = get_linklist0(cur_c);
        else
            ll_cur = get_linklist(cur_c, level);

        if (*ll_cur && !isUpdate) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        setListCount(ll_cur, selectedNeighbors.size());
        tableint* data = (tableint*)(ll_cur + 1);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            if (data[idx] && !isUpdate) throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            data[idx] = selectedNeighbors[idx];
        }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

        linklistsizeint* ll_other;
        // TODO: Replace
        if (level == 0)
            ll_other = get_linklist0(selectedNeighbors[idx]);
        else
            ll_other = get_linklist(selectedNeighbors[idx], level);

        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbors[idx] == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbors[idx]])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        tableint* data = (tableint*)(ll_other + 1);

        bool is_cur_c_present = false;
        if (isUpdate) {
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
                }
            }
        }

        // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]`
        // then no need to modify any connections or run the heuristics.
        if (!is_cur_c_present) {
            if (sz_link_list_other < Mcurmax) {
                data[sz_link_list_other] = cur_c;
                setListCount(ll_other, sz_link_list_other + 1);
            } else {
                // finding the "weakest" element to replace it with the new one
                dist_t d_max =
                    fstdistfunc_(getDataByInternalId(cur_c),
                                 getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
                // Heuristic:
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                    candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(
                        fstdistfunc_(getDataByInternalId(data[j]),
                                     getDataByInternalId(selectedNeighbors[idx]), dist_func_param_),
                        data[j]);
                }

                getNeighborsByHeuristic2(candidates, Mcurmax);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }

                setListCount(ll_other, indx);
                // Nearest K:
                /*int indx = -1;
                for (int j = 0; j < sz_link_list_other; j++) {
                    dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
                getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) { indx = j; d_max =
                d;
                    }
                }
                if (indx >= 0) {
                    data[indx] = cur_c;
                } */
            }
        }
    }

    return next_closest_entry_point;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::repairConnectionsForUpdate(const void* dataPoint,
                                                         tableint entryPointInternalId,
                                                         tableint dataPointInternalId,
                                                         int dataPointLevel, int maxLevel) {
    tableint currObj = entryPointInternalId;
    if (dataPointLevel < maxLevel) {
        dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
        for (int level = maxLevel; level > dataPointLevel; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int* data;
                std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                data = get_linklist_at_level(currObj, level);
                int size = getListCount(data);
                tableint* datal = (tableint*)(data + 1);

                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
    }

    if (dataPointLevel > maxLevel)
        throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

    for (int level = dataPointLevel; level >= 0; level--) {
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            topCandidates = searchBaseLayer(currObj, dataPoint, level);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            filteredTopCandidates;
        while (topCandidates.size() > 0) {
            if (topCandidates.top().second != dataPointInternalId)
                filteredTopCandidates.push(topCandidates.top());

            topCandidates.pop();
        }

        // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where
        // `topCandidates` could just contains entry point itself. To prevent self loops, the
        // `topCandidates` is filtered and thus can be empty.
        if (filteredTopCandidates.size() > 0) {
            bool epDeleted = isMarkedDeleted(entryPointInternalId);
            if (epDeleted) {
                filteredTopCandidates.emplace(
                    fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId),
                                 dist_func_param_),
                    entryPointInternalId);
                if (filteredTopCandidates.size() > ef_construction_) filteredTopCandidates.pop();
            }

            currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId,
                                                filteredTopCandidates, level, true);
        }
    }
}
//--------------------------------------------------------------------------------------------------
// TODO: Investigate in Paper why this is done. Maybe to reduce local minima?
// -> To avoid unconnected clusters
template <typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByHeuristic2(
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>& top_candidates,
    const size_t M) {
    if (top_candidates.size() < M) {
        return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (!queue_closest.empty()) {
        if (return_list.size() >= M) break;

        std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        dist_t dist_to_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<dist_t, tableint> second_pair : return_list) {
            dist_t curdist =
                fstdistfunc_(getDataByInternalId(second_pair.second),
                             getDataByInternalId(curent_pair.second), dist_func_param_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }

        if (good) {
            return_list.push_back(curent_pair);
        }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
        top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
}

//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::updatePoint(const void* dataPoint, tableint internalId,
                                          float updateNeighborProbability) {
    // update the feature vector associated with existing point with new vector
    memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

    int maxLevelCopy = maxlevel_;
    tableint entryPointCopy = enterpoint_node_;

    // If point to be updated is entry point and graph just contains single element then just
    // return.
    if (entryPointCopy == internalId && cur_element_count == 1) return;

    int elemLevel = element_levels_[internalId];
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int layer = 0; layer <= elemLevel; layer++) {
        // Candidates are maximal 2 hops away.
        std::unordered_set<tableint> sCand;
        std::unordered_set<tableint> sNeigh;

        // get direct neighbors
        std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);

        if (listOneHop.size() == 0) continue;

        sCand.insert(internalId);

        for (auto&& elOneHop : listOneHop) {
            sCand.insert(elOneHop);

            if (distribution(update_probability_generator_) > updateNeighborProbability) continue;

            sNeigh.insert(elOneHop);

            std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
            for (auto&& elTwoHop : listTwoHop) {
                sCand.insert(elTwoHop);
            }
        }

        // get
        for (auto&& neigh : sNeigh) {
            // if (neigh == internalId)
            //     continue;

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                candidates;

            // sCand guaranteed to have size >= 1 (because internalId inside)
            size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;

            size_t elementsToKeep = std::min(ef_construction_, size);
            for (auto&& cand : sCand) {
                if (cand == neigh) continue;

                dist_t distance = fstdistfunc_(getDataByInternalId(neigh),
                                               getDataByInternalId(cand), dist_func_param_);

                if (candidates.size() < elementsToKeep) {
                    candidates.emplace(distance, cand);
                } else {
                    if (distance < candidates.top().first) {
                        candidates.pop();
                        candidates.emplace(distance, cand);
                    }
                }
            }

            // Retrieve neighbours using heuristic.
            getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

            // Set connections from neigh to candidates.
            {
                std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                linklistsizeint* ll_cur;
                ll_cur = get_linklist_at_level(neigh, layer);
                size_t candSize = candidates.size();
                setListCount(ll_cur, candSize);
                tableint* data = (tableint*)(ll_cur + 1);
                for (size_t idx = 0; idx < candSize; idx++) {
                    data[idx] = candidates.top().second;
                    candidates.pop();
                }
            }
        }
    }

    repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::addPoint(const void* data_point, labeltype label,
                                       bool replace_deleted) {
    if (!allow_replace_deleted_ && replace_deleted) {
        throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
    }

    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
    if (!replace_deleted) {
        addPoint(data_point, label, -1);
        return;
    }

    // check if there is vacant place
    tableint internal_id_replaced;
    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
    bool is_vacant_place = !deleted_elements.empty();
    if (is_vacant_place) {
        internal_id_replaced = *deleted_elements.begin();
        deleted_elements.erase(internal_id_replaced);
    }
    lock_deleted_elements.unlock();

    // if there is no vacant place then add or update point
    // else add point to vacant place
    if (!is_vacant_place) {
        addPoint(data_point, label, -1);
    } else {
        // we assume that there are no concurrent operations on deleted element
        labeltype label_replaced = getExternalLabel(internal_id_replaced);
        setExternalLabel(internal_id_replaced, label);

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        label_lookup_.erase(label_replaced);
        label_lookup_[label] = internal_id_replaced;
        lock_table.unlock();

        unmarkDeletedInternal(internal_id_replaced);
        updatePoint(data_point, internal_id_replaced, 1.0);
    }
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
int HierarchicalNSW<dist_t>::getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
tableint HierarchicalNSW<dist_t>::addPoint(const void* data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
        // Checking if the element with the same label already exists
        // if so, updating it *instead* of creating a new element.
        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search != label_lookup_.end()) {
            tableint existingInternalId = search->second;
            if (allow_replace_deleted_) {
                if (isMarkedDeleted(existingInternalId)) {
                    throw std::runtime_error(
                        "Can't use addPoint to update deleted elements if replacement of deleted "
                        "elements is enabled.");
                }
            }
            lock_table.unlock();

            if (isMarkedDeleted(existingInternalId)) {
                unmarkDeletedInternal(existingInternalId);
            }
            updatePoint(data_point, existingInternalId, 1.0);

            return existingInternalId;
        }

        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }

        cur_c = cur_element_count;
        cur_element_count++;
        label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0) curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0,
           size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (curlevel) {
        linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel + 1);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
        if (curlevel < maxlevelcopy) {
            dist_t curdist =
                fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxlevelcopy; level > curlevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int* data;
                    std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist(currObj, level);
                    int size = getListCount(data);

                    tableint* datal = (tableint*)(data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d =
                            fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        bool epDeleted = isMarkedDeleted(enterpoint_copy);
        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            if (level > maxlevelcopy || level < 0)  // possible?
                throw std::runtime_error("Level error");

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                top_candidates = searchBaseLayer(currObj, data_point, level);
            if (epDeleted) {
                top_candidates.emplace(
                    fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy),
                                 dist_func_param_),
                    enterpoint_copy);
                if (top_candidates.size() > ef_construction_) top_candidates.pop();
            }
            currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
        }
    } else {
        // Do nothing for the first element
        enterpoint_node_ = 0;
        maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    return cur_c;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::unmarkDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (isMarkedDeleted(internalId)) {
        unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        *ll_cur &= ~DELETE_MARK;
        num_deleted_ -= 1;
        if (allow_replace_deleted_) {
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
            deleted_elements.erase(internalId);
        }
    } else {
        throw std::runtime_error("The requested to undelete element is not deleted");
    }
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
template <bool bare_bone_search, bool collect_metrics>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayerST(tableint ep_id, const void* data_point, size_t ef,
                                           BaseFilterFunctor* isIdAllowed,
                                           BaseSearchStopCondition<dist_t>* stop_condition) const {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidate_set;

    dist_t lowerBound;
    if (bare_bone_search ||
        (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
        char* ep_data = getDataByInternalId(ep_id);
        dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);
        if (!bare_bone_search && stop_condition) {
            stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
        }
        candidate_set.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
        std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
        dist_t candidate_dist = -current_node_pair.first;

        bool flag_stop_search;
        if (bare_bone_search) {
            flag_stop_search = candidate_dist > lowerBound;
        } else {
            if (stop_condition) {
                flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
            } else {
                flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
            }
        }
        if (flag_stop_search) {
            break;
        }
        candidate_set.pop();

        tableint current_node_id = current_node_pair.second;
        int* data = (int*)get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
        if (collect_metrics) {
            metric_hops++;
            metric_distance_computations += size;
        }

        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            //                    if (candidate_id == 0) continue;

            if (!(visited_array[candidate_id] == visited_array_tag)) {
                visited_array[candidate_id] = visited_array_tag;

                char* currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                bool flag_consider_candidate;
                if (!bare_bone_search && stop_condition) {
                    flag_consider_candidate =
                        stop_condition->should_consider_candidate(dist, lowerBound);
                } else {
                    flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                }

                if (flag_consider_candidate) {
                    candidate_set.emplace(-dist, candidate_id);
                    if (bare_bone_search ||
                        (!isMarkedDeleted(candidate_id) &&
                         ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                        top_candidates.emplace(dist, candidate_id);
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->add_point_to_result(getExternalLabel(candidate_id),
                                                                currObj1, dist);
                        }
                    }

                    bool flag_remove_extra = false;
                    if (!bare_bone_search && stop_condition) {
                        flag_remove_extra = stop_condition->should_remove_extra();
                    } else {
                        flag_remove_extra = top_candidates.size() > ef;
                    }
                    while (flag_remove_extra) {
                        tableint id = top_candidates.top().second;
                        top_candidates.pop();
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->remove_point_from_result(getExternalLabel(id),
                                                                     getDataByInternalId(id), dist);
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                    }

                    if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                }
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::priority_queue<std::pair<dist_t, labeltype>> HierarchicalNSW<dist_t>::searchKnn(
    const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist =
        fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    // search in upper levels
    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data;

            data = (unsigned int*)get_linklist(currObj, level);
            int size = getListCount(data);
            metric_hops++;
            metric_distance_computations += size;

            // find the closest candidate among the neighbors
            tableint* datal = (tableint*)(data + 1);
            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    // level 0 search
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    if (bare_bone_search) {
        top_candidates =
            searchBaseLayerST<true>(currObj, query_data, std::max(ef_, k), isIdAllowed);
    } else {
        top_candidates =
            searchBaseLayerST<false>(currObj, query_data, std::max(ef_, k), isIdAllowed);
    }

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        std::pair<dist_t, tableint> rez = top_candidates.top();
        result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
        top_candidates.pop();
    }
    return result;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::saveIndex(const std::string& location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
        unsigned int linkListSize =
            element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize) output.write(linkLists_[i], linkListSize);
    }
    output.close();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
inline labeltype HierarchicalNSW<dist_t>::getExternalLabel(
    tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label,
           (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
           sizeof(labeltype));
    return return_label;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
linklistsizeint* HierarchicalNSW<dist_t>::get_linklist_at_level(tableint internal_id,
                                                                int level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
linklistsizeint* HierarchicalNSW<dist_t>::get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint*)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
linklistsizeint* HierarchicalNSW<dist_t>::get_linklist0(tableint internal_id,
                                                        char* data_level0_memory) const {
    return (linklistsizeint*)(data_level0_memory + internal_id * size_data_per_element_ +
                              offsetLevel0_);
}
//--------------------------------------------------------------------------------------------------
// TODO: Replace this function
template <typename dist_t>
linklistsizeint* HierarchicalNSW<dist_t>::get_linklist0(tableint internal_id) const {
    return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ +
                              offsetLevel0_);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
bool HierarchicalNSW<dist_t>::isMarkedDeleted(vectorlib::hnsw::tableint internalId) const {
    unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
    return *ll_cur & DELETE_MARK;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::loadIndex(const std::string& location, SpaceInterface<dist_t>* s,
                                        size_t max_elements_i) {
    // TODO: Investigate Layout of data (do little graphs!)
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    clear();

    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    // max_elements_ cannot be smaller than our current elements count
    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count) max_elements = max_elements_;
    max_elements_ = max_elements;

    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    // Space Interface stuff
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    auto pos = input.tellg();

    /// Optional - check if index is ok:
    // TODO: Check what this really does
    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {
        if (input.tellg() < 0 || input.tellg() >= total_filesize) {
            throw std::runtime_error(
                "Index seems to be corrupted or unsupported");
        }

        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize != 0) {
            input.seekg(linkListSize, input.cur);
        }
    }

    // throw exception if it either corrupted or old index
    if (input.tellg() != total_filesize)
        throw std::runtime_error("Index seems to be corrupted or unsupported");

    input.clear();
    /// Optional check end

    // Set back read position after check
    input.seekg(pos, input.beg);

    // Allocate whole memory we need for index, even if we don't currently have
    // as many elements
    data_level0_memory_ = (char*)malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    // move mutexes into the right variables
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    // Load elements
    linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
    if (linkLists_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: loadIndex failed to allocate linklists");
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
        label_lookup_[getExternalLabel(i)] = i;
        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize == 0) {
            element_levels_[i] = 0;
            linkLists_[i] = nullptr;
        } else {
            element_levels_[i] = linkListSize / size_links_per_element_;
            linkLists_[i] = (char*)malloc(linkListSize);
            if (linkLists_[i] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist");
            input.read(linkLists_[i], linkListSize);
        }
    }

    for (size_t i = 0; i < cur_element_count; i++) {
        if (isMarkedDeleted(i)) {
            num_deleted_ += 1;
            if (allow_replace_deleted_) deleted_elements.insert(i);
        }
    }

    input.close();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::clear() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
        if (element_levels_[i] > 0) free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count = 0;
    visited_list_pool_.reset(nullptr);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
struct CompareByFirst {
    constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                              std::pair<dist_t, tableint> const& b) const noexcept {
        return a.first < b.first;
    }
};
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::setEf(size_t ef) {
    ef_ = ef;
}
//--------------------------------------------------------------------------------------------------
// Explicit instantiation
template class HierarchicalNSW<float>;
template class HierarchicalNSW<int>;
}  // namespace hnsw
}  // namespace vectorlib