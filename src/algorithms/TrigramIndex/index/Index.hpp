// ---------------------------------------------------------------------------
#ifndef TRIGRAM_INDEX_INTERFACE_HPP
#define TRIGRAM_INDEX_INTERFACE_HPP
// ---------------------------------------------------------------------------
#include <cstdint>
#include "models/Trigram.hpp"
// ---------------------------------------------------------------------------
template <typename ValueT, typename ContainerT, uint8_t BucketSize>
class Index {
    public:
    /// Destructor.
    virtual ~Index() = default;
    /// Insert a key-value pair into the index.
    virtual void insert(Trigram key, ValueT value) = 0;
    /// Lookup the value for given trigram.
    virtual ContainerT* lookup(Trigram key) = 0;
};
// ---------------------------------------------------------------------------
template <typename ContainerT, uint8_t Size>
struct Bucket {
    /// The containers referenced by the bucket.
    std::array<ContainerT, Size> containers;
    /// The number of chains referenced by the bucket.
    uint8_t getSize() const { return Size; }
};
// ---------------------------------------------------------------------------
#endif // TRIGRAM_INDEX_INTERFACE_HPP