#ifndef TF_IDF_HPP
#define TF_IDF_HPP
//---------------------------------------------------------------------------
#include "scoring_function.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
class TfIdf : public ScoringFunction {
 public:
  /// Constructor.
  explicit TfIdf(uint32_t doc_count);
  /// Calculates the tf-idf score for a given document and word.
  double score(const DocStats& doc_stats, const WordStats& word_stats) const override;

 private:
  /// The total number of documents.
  const uint32_t doc_count;
};
//---------------------------------------------------------------------------
}  // namespace scoring
//---------------------------------------------------------------------------
#endif  // TF_IDF_HPP