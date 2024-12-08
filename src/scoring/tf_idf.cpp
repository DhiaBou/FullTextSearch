#include "tf_idf.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
TfIdf::TfIdf(uint32_t doc_count) : doc_count(doc_count) {}
//---------------------------------------------------------------------------
double TfIdf::score(const DocStats& doc_stats, const QueryStats& query_stats) {
  double result = 0;
  for (const auto& word : query_stats.query_words) {
    result += (word.frequency / doc_stats.doc_length) * idf(doc_count, word.total_count);
  }
  return result;
}
//---------------------------------------------------------------------------
}  // namespace scoring