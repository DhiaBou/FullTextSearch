
#include "stemmingtokenizer.hpp"

#include "snowball/api.h"
#include "tokenizer_rules.hpp"
namespace tokenizer {

StemmingTokenizer::StemmingTokenizer(const char *data, const size_t size)
    : data_(data), size_(size), currentPos_(0) {
  // Create the Snowball stemmer environment once
  stemEnv_ = english_UTF_8_create_env();
}

StemmingTokenizer::~StemmingTokenizer() {
  // Close the Snowball environment
  english_UTF_8_close_env(stemEnv_);
}

void StemmingTokenizer::skipDelimiters() {
  while (currentPos_ < size_ && isDelimiter(data_[currentPos_])) {
    ++currentPos_;
  }
}

std::string StemmingTokenizer::nextToken(bool skip_stop_words) {
  skipDelimiters();

  if (currentPos_ >= size_) {
    return "";
  }

  // Find the start of the token
  size_t tokenStart = currentPos_;

  // Advance until we hit a delimiter or end of data
  while (currentPos_ < size_ && !isDelimiter(data_[currentPos_])) {
    ++currentPos_;
  }

  size_t tokenLength = currentPos_ - tokenStart;

  // Extract and convert to lowercase directly
  std::string token;
  token.reserve(tokenLength);
  for (size_t i = tokenStart; i < tokenStart + tokenLength; ++i) {
    token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(data_[i]))));
  }

  if (skip_stop_words && isStopWord(token)) {
    return nextToken(skip_stop_words);
  }

  // Stem the token using Snowball
  SN_set_current(stemEnv_, (int)token.size(),
                 reinterpret_cast<const unsigned char *>(token.c_str()));

  // Perform the stemming
  english_UTF_8_stem(stemEnv_);
  stemEnv_->p[stemEnv_->l] = '\0';
  return std::string(reinterpret_cast<char *>(stemEnv_->p), stemEnv_->l);
}

}  // namespace tokenizer
