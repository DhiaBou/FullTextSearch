#ifndef TF_IDF_HPP
#define TF_IDF_HPP
//---------------------------------------------------------------------------
#include "ranking.hpp"
//---------------------------------------------------------------------------
namespace ranking {
//---------------------------------------------------------------------------
class TfIdf : public Ranking {
 public:
  /// Constructor.
  explicit TfIdf(uint32_t doc_count);
  /// Calculates the BM25 score for a given document and query.
  double score(const DocStats& doc_stats, const QueryStats& query_stats) override;

 private:
  const uint32_t doc_count;
};
//---------------------------------------------------------------------------
}  // namespace ranking
//---------------------------------------------------------------------------
#endif  // TF_IDF_HPP