#ifndef RANKING_HPP
#define RANKING_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <vector>
//---------------------------------------------------------------------------
namespace ranking {
//---------------------------------------------------------------------------
struct QueryStats {
  struct QueryWordStats {
    /// The number of times the query word appears in the document.
    uint32_t frequency;
    /// The number of documents the query word appears in.
    uint32_t total_count;
  };
  /// Statistics for each word in the query.
  std::vector<QueryWordStats> query_words;
};
//---------------------------------------------------------------------------
struct DocStats {
  /// The associated document's ID.
  uint32_t doc_id;
  /// The associated document's length in words.
  uint32_t doc_length;
};
//---------------------------------------------------------------------------
class Ranking {
 public:
  /// Destructor.
  virtual ~Ranking() = default;
  /// Calculates a score for a given document and query.
  virtual double score(const DocStats& doc_stats, const QueryStats& query_stats) = 0;
};
//---------------------------------------------------------------------------
/**
 * Calculates the inverse document frequency.
 *
 * @param doc_count The total number of documents.
 * @param doc_frequency The number of documents containing the term.
 * @return The IDF value.
 */
inline double idf(uint32_t doc_count, uint32_t doc_frequency) {
  return std::log((doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1);
}
//---------------------------------------------------------------------------
}  // namespace ranking
//---------------------------------------------------------------------------
#endif  // RANKING_HPP