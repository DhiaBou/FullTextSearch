#include "tf_idf.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
TfIdf::TfIdf(uint32_t doc_count) : doc_count(doc_count) {}
//---------------------------------------------------------------------------
double TfIdf::score(const DocStats& doc_stats, const QueryStats& query_stats) const {
  double result = 0;
  for (const auto& word : query_stats.query_words) {
    result += (static_cast<double>(word.frequency) / static_cast<double>(doc_stats.doc_length)) *
              idf(doc_count, word.total_count);
  }
  return result;
}
//---------------------------------------------------------------------------
}  // namespace scoring