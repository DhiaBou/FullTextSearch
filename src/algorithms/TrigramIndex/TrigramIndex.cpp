// ---------------------------------------------------------------------------
#include "TrigramIndex.hpp"
//---------------------------------------------------------------------------
std::vector<Appearance>* TrigramIndex::lookup(Trigram trigram) {
    auto it = index.find(trigram.getValue());
    if (it != index.end()) {
        return &it->second;
    } else {
        return nullptr;
    }
}
//---------------------------------------------------------------------------
void TrigramIndex::insert(Appearance app) {
    index[app.trigram.getValue()].push_back(app);
}
//---------------------------------------------------------------------------