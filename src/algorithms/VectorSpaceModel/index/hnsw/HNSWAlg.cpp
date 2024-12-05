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
HierarchicalNSW<dist_t>::HierarchicalNSW(
                            SpaceInterface<dist_t> *s,
                            const std::string& location,
                           // bool nmslib,
                            size_t max_elements,
                            bool allow_replace_deleted)
                            : allow_replace_deleted_(allow_replace_deleted)
{
    loadIndex(location, s, max_elements);
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::HierarchicalNSW(SpaceInterface<dist_t>* s,
                                         size_t max_elements, size_t M,
                                         size_t ef_construction, size_t random_seed,
                                         bool allow_replace_deleted)
{
    // TODO
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
HierarchicalNSW<dist_t>::~HierarchicalNSW()
{
    // TODO
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::addPoint(const void *datapoint, labeltype label, bool replaceDeleted)
{
    // TODO
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::priority_queue<std::pair<dist_t, labeltype >>
HierarchicalNSW<dist_t>::searchKnn(const void* queryData, size_t k,
                                   BaseFilterFunctor* isIdAllowed) const
{
    // TODO

    return std::priority_queue<std::pair<dist_t, labeltype >>();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
HierarchicalNSW<dist_t>::searchKnnCloserFirst(const void* queryData, size_t k,
                                              BaseFilterFunctor* isIdAllowed) const
{
    // TODO
    return std::vector<std::pair<dist_t, labeltype>>();
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::saveIndex(const std::string& location)
{
    // TODO
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
inline labeltype HierarchicalNSW<dist_t>::getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
    return return_label;
}
//--------------------------------------------------------------------------------------------------
template <typename dist_t>
void HierarchicalNSW<dist_t>::loadIndex(const std::string& location, SpaceInterface<dist_t>* s,
                                 size_t max_elements_i)
{
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

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
    if (max_elements < cur_element_count)
        max_elements = max_elements_;
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
            throw std::runtime_error("Index seems to be corrupted or unsupported");
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

    // Allocate whole memory we need for index, even if we don't currently have as many elements
    data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    // move mutexes into the right variables
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    // Load elements
    linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
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
            linkLists_[i] = (char *) malloc(linkListSize);
            if (linkLists_[i] == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
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
void HierarchicalNSW<dist_t>::clear()
{
    // TODO
}
//--------------------------------------------------------------------------------------------------
// Explicit instantiation
template class HierarchicalNSW<float>;
template class HierarchicalNSW<int>;
}
}