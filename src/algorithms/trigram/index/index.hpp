#ifndef TRIGRAM_INDEX_INTERFACE_HPP
#define TRIGRAM_INDEX_INTERFACE_HPP
//---------------------------------------------------------------------------
#include <fstream>
//---------------------------------------------------------------------------
#include "models/trigram.hpp"
//---------------------------------------------------------------------------
namespace trigramlib {
//---------------------------------------------------------------------------
template <typename ValueT, typename ContainerT, uint8_t BucketSize>
class Index {
 public:
  /// Destructor.
  virtual ~Index() = default;
  /// Insert a key-value pair into the index.
  virtual void insert(Trigram key, ValueT value) = 0;
  /// Lookup the value for given trigram.
  virtual ContainerT* lookup(Trigram key) = 0;
  /// Write the underlying data structure to specified file.
  virtual void store(std::ofstream& file) = 0;
  /// Load the underlying data structure from given data.
  virtual void load(const char* begin, const char* end) = 0;
  /// Determine the memory footprint of the index in Bytes.
  virtual uint64_t footprint() = 0;
};
//---------------------------------------------------------------------------
template <typename ContainerT, uint8_t Size>
struct Bucket {
  /// The containers referenced by the bucket.
  std::array<ContainerT, Size> containers;
  /// The number of chains referenced by the bucket.
  uint8_t getSize() const { return Size; }
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_INDEX_INTERFACE_HPP