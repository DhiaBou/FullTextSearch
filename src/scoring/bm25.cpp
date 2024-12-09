#include "bm25.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
BM25::BM25(uint32_t doc_count, double avg_doc_length, double k1, double b)
    : doc_count(doc_count), avg_doc_length(avg_doc_length), k1(k1), b(b) {}
//---------------------------------------------------------------------------
double BM25::score(const DocStats& doc_stats, const QueryStats& query_stats) const {
  double result = 0;
  for (const auto& word : query_stats.query_words) {
    result += idf(doc_count, word.total_count) *
              ((static_cast<double>(word.frequency) * (k1 + 1)) /
               (static_cast<double>(word.frequency) +
                (k1 * (1 - b + b * (static_cast<double>(doc_stats.doc_length) / avg_doc_length)))));
  }
  return result;
}
//---------------------------------------------------------------------------
}  // namespace scoring
