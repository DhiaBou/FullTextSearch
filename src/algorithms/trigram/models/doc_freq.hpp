#ifndef TRIGRAM_DOCFREQ_HPP
#define TRIGRAM_DOCFREQ_HPP
//---------------------------------------------------------------------------
#include <cstdint>
#include <functional>
//---------------------------------------------------------------------------
namespace trigramlib {
//---------------------------------------------------------------------------
/**
 * A container to store the appearance of a trigram in a document and its
 * number of appearances in this document.
 */
struct DocFreq {
  /// The document's document-ID.
  uint32_t doc_id;
  /// The number of appearances of a Trigram.
  uint32_t freq;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_DOCFREQ_HPP