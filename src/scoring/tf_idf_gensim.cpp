
#include "tf_idf_gensim.hpp"

#include <cmath>
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
TfIdfGensim::TfIdfGensim(uint32_t doc_count) : doc_count(doc_count) {}
//---------------------------------------------------------------------------
double TfIdfGensim::score(const DocStats& doc_stats, const WordStats& word_stats) const {
  return (static_cast<double>(word_stats.frequency) / static_cast<double>(doc_stats.doc_length)) *
         std::log2(static_cast<double>(doc_count) / word_stats.total_count);
}
//---------------------------------------------------------------------------
double TfIdfGensim::score(const DocStats& doc_stats, const WordStats& word_stats,
                          double idf) const {
  return (static_cast<double>(word_stats.frequency) / static_cast<double>(doc_stats.doc_length)) *
         std::log2(doc_count / word_stats.total_count);
}
//---------------------------------------------------------------------------
}  // namespace scoring