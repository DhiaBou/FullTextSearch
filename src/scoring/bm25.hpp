#ifndef BM25_HPP
#define BM25_HPP
//---------------------------------------------------------------------------
#include "scoring_function.hpp"
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
class BM25 : public ScoringFunction {
 public:
  /// Constructor.
  BM25(uint32_t doc_count, double avg_doc_length, double k1, double b);
  /// Calculates the BM25 score for a given document and query.
  double score(const DocStats& doc_stats, const QueryStats& query_stats) const override;

 private:
  /// The total number of documents.
  const uint32_t doc_count;
  /// The average length of documents in words.
  const double avg_doc_length;
  /// A constant.
  const double k1;
  /// A scaling factor.
  const double b;
};
//---------------------------------------------------------------------------
}  // namespace scoring
//---------------------------------------------------------------------------
#endif  // BM25_HPP