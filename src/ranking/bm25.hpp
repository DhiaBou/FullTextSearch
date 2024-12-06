#ifndef BM25_HPP
#define BM25_HPP
//---------------------------------------------------------------------------
#include "ranking.hpp"
//---------------------------------------------------------------------------
namespace ranking {
//---------------------------------------------------------------------------
class BM25 : public Ranking {
 public:
  /// Constructor.
  BM25(uint32_t doc_count, uint32_t avg_doc_length, double k1, double b);
  /// Calculates the BM25 score for a given document and query.
  double score(const DocStats& doc_stats, const QueryStats& query_stats) override;

 private:
  const uint32_t doc_count;
  const uint32_t avg_doc_length;
  const double k1;
  const double b;
};
//---------------------------------------------------------------------------
}  // namespace ranking
//---------------------------------------------------------------------------
#endif  // BM25_HPP