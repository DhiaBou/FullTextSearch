#include "bm25.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
BM25::BM25(uint32_t doc_count, double avg_doc_length, double k1, double b)
    : doc_count(doc_count), avg_doc_length(avg_doc_length), k1(k1), b(b) {}
//---------------------------------------------------------------------------
double BM25::score(const DocStats& doc_stats, const WordStats& word_stats) const {
  return idf(doc_count, word_stats.total_count) *
         ((static_cast<double>(word_stats.frequency) * (k1 + 1.0)) /
          (static_cast<double>(word_stats.frequency) +
           (k1 * (1.0 - b + b * (static_cast<double>(doc_stats.doc_length) / avg_doc_length)))));
}
//---------------------------------------------------------------------------
}  // namespace scoring
