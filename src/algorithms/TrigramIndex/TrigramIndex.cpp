// ---------------------------------------------------------------------------
#include "TrigramIndex.hpp"
//---------------------------------------------------------------------------
std::vector<Appearance>* TrigramIndex::lookup(Trigram trigram) {
    auto it = index.find(trigram.getRawValue());
    if (it != index.end()) {
        return &it->second;
    } else {
        return nullptr;
    }
}
//---------------------------------------------------------------------------
void TrigramIndex::insert(Appearance app) {
    index[app.trigram.getRawValue()].push_back(app);
}
//---------------------------------------------------------------------------