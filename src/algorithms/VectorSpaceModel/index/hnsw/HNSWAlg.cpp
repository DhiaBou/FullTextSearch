//
// Created by miguel on 05.12.24.
//

#include "HNSWAlg.h"

#include <fstream>

#include "../VectorSpaceLib.h"
#include "VisitedList.h"

namespace vectorlib {
namespace hnsw {
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t>* s,
                                         const std::string& location,
                                         // bool nmslib,
                                         size_t max_elements
                                         // bool allow_replace_deleted
                                         )
// allow_replace_deleted_(allow_replace_deleted)
{
    loadIndex(location, s, max_elements);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t>* s,
                                         size_t max_elements, size_t M,
                                         size_t ef_construction,
                                         size_t random_seed
                                         // bool allow_replace_deleted
                                         )
    : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
      link_list_locks_(max_elements),
      element_levels_(max_elements)
// allow_replace_deleted_(allow_replace_deleted)
{
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
        std::cout
            << "Cap to 10000 will be applied for the rest of the processing.\n";
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
    size_data_per_element_ =
        size_links_level0_ + data_size_ + sizeof(labeltype);

    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory");

    cur_element_count = 0;

    visited_list_pool_ =
        std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error(
            "Not enough memory: HierarchicalNSW failed to allocate linklists");
    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
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
void HierarchicalNSW<dist_t>::addPoint(const void* datapoint, labeltype label,
                                       bool replaceDeleted) {
    // TODO
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::priority_queue<std::pair<dist_t, labeltype>>
HierarchicalNSW<dist_t>::searchKnn(const void* queryData, size_t k,
                                   BaseFilterFunctor* isIdAllowed) const {
    // TODO

    return std::priority_queue<std::pair<dist_t, labeltype>>();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
HierarchicalNSW<dist_t>::searchKnnCloserFirst(
    const void* queryData, size_t k, BaseFilterFunctor* isIdAllowed) const {
    // TODO
    return std::vector<std::pair<dist_t, labeltype>>();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::saveIndex(const std::string& location) {
    // TODO
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
inline labeltype HierarchicalNSW<dist_t>::getExternalLabel(
    tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label,
           (data_level0_memory_ + internal_id * size_data_per_element_ +
            label_offset_),
           sizeof(labeltype));
    return return_label;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::loadIndex(const std::string& location,
                                        SpaceInterface<dist_t>* s,
                                        size_t max_elements_i) {
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

    /*
        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }
    */

    input.close();

    return;
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
    constexpr bool operator()(
        std::pair<dist_t, tableint> const& a,
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