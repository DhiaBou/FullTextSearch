#include "bm25.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
BM25::BM25(uint32_t doc_count, uint32_t avg_doc_length, double k1, double b)
    : doc_count(doc_count), avg_doc_length(avg_doc_length), k1(k1), b(b) {}
//---------------------------------------------------------------------------
double BM25::score(const DocStats& doc_stats, const QueryStats& query_stats) const {
  double result = 0;
  for (const auto& word : query_stats.query_words) {
    result += idf(doc_count, word.total_count) *
              ((word.frequency * (k1 + 1)) /
               (word.frequency + (k1 * (1 - b + b * (doc_stats.doc_length / avg_doc_length)))));
  }
  return result;
}
//---------------------------------------------------------------------------
}  // namespace scoring
