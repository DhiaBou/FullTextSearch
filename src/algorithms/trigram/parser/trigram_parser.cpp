#include <cassert>
//---------------------------------------------------------------------------
#include "trigram_parser.hpp"
//---------------------------------------------------------------------------
namespace trigramlib {
//---------------------------------------------------------------------------
TrigramParser::TrigramParser(const char* begin, const char* end)
    : end(end), it(begin), word_begin(begin), trigram_begin(begin), offset(0) {}
//---------------------------------------------------------------------------
bool TrigramParser::hasNext() {
  bool found = false;

  while (it < end) {
    auto c = static_cast<unsigned char>(*it);
    if (c < 128 && kWhiteList[c]) {
      if (it - trigram_begin >= 2) {
        // three consecutive white-listed ASCII characters
        assert(it - trigram_begin == 2);
        assert(trigram_begin >= word_begin);

        found = true;

        trigram[0] = static_cast<char>(std::tolower(*trigram_begin));
        trigram[1] = static_cast<char>(std::tolower(*(trigram_begin + 1)));
        trigram[2] = static_cast<char>(std::tolower(*(trigram_begin + 2)));
        offset = static_cast<uint8_t>(trigram_begin - word_begin);

        ++trigram_begin;
      }
    } else {
      if (it - word_begin == 2) {
        // stand-alone two-character "trigram"
        found = true;

        trigram[0] = static_cast<char>(std::tolower(*word_begin));
        trigram[1] = static_cast<char>(std::tolower(*(word_begin + 1)));
        trigram[2] = '\0';
        offset = 0;
      }

      trigram_begin = it + 1;
      word_begin = it + 1;
    }
    ++it;

    if (found) return true;
  }

  return false;
}
//---------------------------------------------------------------------------
Trigram TrigramParser::next() const { return {trigram, offset}; }
//---------------------------------------------------------------------------
}  // namespace trigramlib