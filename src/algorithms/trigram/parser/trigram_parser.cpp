#include <cassert>
//---------------------------------------------------------------------------
#include "trigram_parser.hpp"
//---------------------------------------------------------------------------
TrigramParser::TrigramParser(const char* begin, const char* end)
    : end(end), it(begin), word_begin(begin), trigram_begin(begin), trigram(0) {}
//---------------------------------------------------------------------------
bool TrigramParser::hasNext() {
  bool found = false;

  while (it < end) {
    auto c = static_cast<unsigned char>(*it);
    if (c < 128 && white_list[c]) {
      if (it - trigram_begin >= 2) {
        // three consecutive white-listed ASCII characters
        assert(it - trigram_begin == 2);
        assert(trigram_begin >= word_begin);

        auto offset = trigram_begin - word_begin;
        trigram = Trigram(trigram_begin, static_cast<uint8_t>(offset));
        found = true;

        ++trigram_begin;
      }
    } else {
      if (it - word_begin == 2) {
        // stand-alone two-character "trigram"
        char data[3] = {*word_begin, *(word_begin + 1), '\0'};
        trigram = Trigram(data, 0);
        found = true;
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
Trigram TrigramParser::next() const { return trigram; }