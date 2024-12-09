#ifndef SCORING_FUNCTION_HPP
#define SCORING_FUNCTION_HPP
//---------------------------------------------------------------------------
#include <cmath>
#include <cstdint>
#include <vector>
//---------------------------------------------------------------------------
namespace scoring {
//---------------------------------------------------------------------------
/**
 * Wraps statistics for a query word.
 */
struct QueryWordStats {
  /// The number of times the query word appears in the document.
  uint32_t frequency;
  /// The number of documents the query word appears in.
  uint32_t total_count;
};
//---------------------------------------------------------------------------
/**
 * Wraps statistics for a query.
 */
struct QueryStats {
  /// Statistics for each word in the query.
  std::vector<QueryWordStats> query_words;
};
//---------------------------------------------------------------------------
/**
 * Wraps statistics for a document.
 */
struct DocStats {
  /// The associated document's ID.
  uint32_t doc_id;
  /// The associated document's length in words.
  uint32_t doc_length;
};
//---------------------------------------------------------------------------
/**
 * Abstract base class for scoring functions.
 */
class ScoringFunction {
 public:
  /// Destructor.
  virtual ~ScoringFunction() = default;
  /**
   * Calculates a score for a given document and query.
   *
   * @param doc_stats The statistics for the document.
   * @param query_stats The statistics for the query.
   * @return The calculated score for the document and query.
   */
  virtual double score(const DocStats& doc_stats, const QueryStats& query_stats) const = 0;
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
}  // namespace scoring
//---------------------------------------------------------------------------
#endif  // SCORING_FUNCTION_HPP