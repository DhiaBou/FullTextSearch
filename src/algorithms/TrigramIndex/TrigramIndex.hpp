// ---------------------------------------------------------------------------
#ifndef TRIGRAMINDEX_HPP
#define TRIGRAMINDEX_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "Trigram.hpp"
//---------------------------------------------------------------------------
struct Appearance {
    /// The trigram this entry belongs to
    Trigram trigram;
    /// The ID of the document in which the trigram appears
    int docId;
};
//---------------------------------------------------------------------------
class TrigramIndex {
public:
    /// Constructor
    TrigramIndex() = default;
    /// Lookup the trigram in the index
    std::vector<Appearance>* lookup(Trigram trigram);
    /// Insert the appearance in the index
    void insert(Appearance app);
private:
    /// A mapping of trigram to its appearances
    std::unordered_map<uint32_t, std::vector<Appearance>> index;
};
//---------------------------------------------------------------------------
#endif  // TRIGRAMINDEX_HPP
//---------------------------------------------------------------------------