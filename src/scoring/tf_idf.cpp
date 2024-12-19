#include "tf_idf.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
TfIdf::TfIdf(uint32_t doc_count) : doc_count(doc_count) {}
//---------------------------------------------------------------------------
double TfIdf::score(const DocStats& doc_stats, const WordStats& word_stats) const {
  return (static_cast<double>(word_stats.frequency) / static_cast<double>(doc_stats.doc_length)) *
         idf(doc_count, word_stats.total_count);
}
//---------------------------------------------------------------------------
double TfIdf::score(const DocStats& doc_stats, const WordStats& word_stats, double idf) const {
  return (static_cast<double>(word_stats.frequency) / static_cast<double>(doc_stats.doc_length)) * 
  idf;
}
//---------------------------------------------------------------------------
}  // namespace scoring