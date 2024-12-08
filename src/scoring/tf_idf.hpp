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
  /// Calculates the BM25 score for a given document and query.
  double score(const DocStats& doc_stats, const QueryStats& query_stats) override;

 private:
  /// The total number of documents.
  const uint32_t doc_count;
};
//---------------------------------------------------------------------------
}  // namespace scoring
//---------------------------------------------------------------------------
#endif  // TF_IDF_HPP