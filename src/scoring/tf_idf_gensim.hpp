#ifndef TF_IDF_GENSIM_HPP
#define TF_IDF_GENSIM_HPP
//---------------------------------------------------------------------------
#include "scoring_function.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
class TfIdfGensim : public ScoringFunction {
 public:
  /// Constructor.
  explicit TfIdfGensim(uint32_t doc_count);
  /// Calculates the tf-idf score for a given document and word, following the formula of GENSIM
  /// https://radimrehurek.com/gensim_3.8.3/models/tfidfmodel.html.
  double score(const DocStats& doc_stats, const WordStats& word_stats) const override;
  /// Calculates the tf-idf score for a given document and word, following the formula of GENSIM
  /// https://radimrehurek.com/gensim_3.8.3/models/tfidfmodel.html.
  double score(const DocStats& doc_stats, const WordStats& word_stats, double idf) const override;

 private:
  /// The total number of documents.
  const uint32_t doc_count;
};
//---------------------------------------------------------------------------
}  // namespace scoring
//---------------------------------------------------------------------------
#endif  // TF_IDF_GENSIM_HPP