//
// Created by dhia on 12/11/24.
//

#include "simpletokenizer.hpp"

#include "tokenizer_rules.hpp"
namespace tokenizer {

SimpleTokenizer::SimpleTokenizer(const char *data, const size_t size)
    : data_(data), size_(size), currentPos_(0) {}

void SimpleTokenizer::skipDelimiters() {
  while (currentPos_ < size_ && isDelimiter(data_[currentPos_])) {
    ++currentPos_;
  }
}

std::string SimpleTokenizer::nextToken(bool skip_stop_words) {
  skipDelimiters();
  if (currentPos_ >= size_) {
    return "";
  }

  size_t tokenStart = currentPos_;
  while (currentPos_ < size_ && !isDelimiter(data_[currentPos_])) {
    ++currentPos_;
  }

  size_t tokenLength = currentPos_ - tokenStart;
  std::string token;
  token.reserve(tokenLength);
  for (size_t i = tokenStart; i < tokenStart + tokenLength; ++i) {
    token.push_back(static_cast<char>(std::tolower(data_[i])));
  }

  if (skip_stop_words && isStopWord(token)) {
    return nextToken(skip_stop_words);
  }
  return token;
}

}  // namespace tokenizer
