#ifndef TRIGRAM_PARSER_H
#define TRIGRAM_PARSER_H
//---------------------------------------------------------------------------
#include <cstdint>
//---------------------------------------------------------------------------
#include "algorithms/trigram/models/trigram.hpp"
//---------------------------------------------------------------------------
namespace trigramlib {
//---------------------------------------------------------------------------
constexpr std::array<bool, 128> generateWhitelist() {
  std::array<bool, 128> whitelist = {false};
  for (char c = '0'; c <= '9'; ++c) {
    whitelist[static_cast<unsigned char>(c)] = true;
  }
  for (char c = 'A'; c <= 'Z'; ++c) {
    whitelist[static_cast<unsigned char>(c)] = true;
  }
  for (char c = 'a'; c <= 'z'; ++c) {
    whitelist[static_cast<unsigned char>(c)] = true;
  }
  whitelist[static_cast<unsigned char>('$')] = true;
  whitelist[static_cast<unsigned char>('%')] = true;
  whitelist[static_cast<unsigned char>('&')] = true;
  whitelist[static_cast<unsigned char>('+')] = true;
  whitelist[static_cast<unsigned char>('@')] = true;
  return whitelist;
};
//---------------------------------------------------------------------------
constexpr uint32_t kNumPossibleTrigrams = 41 * 41 * 41 * kMaxWordOffset;
//---------------------------------------------------------------------------
class TrigramParser {
 public:
  /// Constructor
  TrigramParser(const char* begin, const char* end);
  /// Whether there is another Trigram.
  bool hasNext();
  /// Get the next trigram.
  Trigram next() const;

 private:
  /// A whitelist of ASCII characters allowed in the trigrams.
  static constexpr std::array<bool, 128> kWhiteList = generateWhitelist();

  /// The end of the text to parse.
  const char* end;
  /// The text iterator.
  const char* it;
  /// The begin of the word the iterator is pointing to.
  const char* word_begin;
  /// The begin of the trigram the iterator is pointing to.
  const char* trigram_begin;
  /// The trigram produced by the hasNext call.
  char trigram[3];
  /// The trigram's offset in the word produced by the hasNext call.
  uint8_t offset;
};
//---------------------------------------------------------------------------
}  // namespace trigramlib
//---------------------------------------------------------------------------
#endif  // TRIGRAM_PARSER_H
