//
// Created by dhia on 12/11/24.
//

#include "simpletokenizer.hpp"

#include "tokenizer_rules.hpp"
namespace tokenizer {

SimpleTokenizer::SimpleTokenizer(const char* data, size_t size)
    : data_(data), size_(size), currentPos_(0) {
  tokenBuffer_.reserve(128);
}

inline void SimpleTokenizer::skipDelimiters() {
  const auto* udata = reinterpret_cast<const unsigned char*>(data_);
  while (currentPos_ < size_ && DELIMS[udata[currentPos_]]) {
    ++currentPos_;
  }
}

std::string SimpleTokenizer::nextToken(bool skip_stop_words) {
  while (true) {
    skipDelimiters();
    if (currentPos_ >= size_) {
      return "";
    }

    const auto* udata = reinterpret_cast<const unsigned char*>(data_);
    size_t tokenStart = currentPos_;

    while (currentPos_ < size_ && !DELIMS[udata[currentPos_]]) {
      ++currentPos_;
    }

    size_t tokenLength = currentPos_ - tokenStart;

    tokenBuffer_.assign(data_ + tokenStart, tokenLength);

    for (char& c : tokenBuffer_) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (skip_stop_words && STOP_WORDS.contains(tokenBuffer_)) {
      continue;
    }

    return tokenBuffer_;
  }
}

}  // namespace tokenizer