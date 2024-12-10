#include "tokenizer.hpp"

#include "snowball/api.h"

static const char DELIMS[] = " \t\n\r.,;:!()[]{}<>?\"'`~@#$%^&*-_=+|\\/";

Tokenizer::Tokenizer(const char *data, size_t size) : data_(data), size_(size), currentPos_(0) {
  // Create the Snowball stemmer environment once
  stemEnv_ = english_UTF_8_create_env();
  for (unsigned char c : DELIMS) {
    delimiters_[c] = true;
  }
}

Tokenizer::~Tokenizer() {
  // Close the Snowball environment
  english_UTF_8_close_env(stemEnv_);
}

void Tokenizer::skipDelimiters() {
  while (currentPos_ < size_ && delimiters_[data_[currentPos_]]) {
    ++currentPos_;
  }
}

std::string Tokenizer::nextToken() {
  skipDelimiters();

  if (currentPos_ >= size_) {
    return "";
  }

  // Find the start of the token
  size_t tokenStart = currentPos_;

  // Advance until we hit a delimiter or end of data
  while (currentPos_ < size_ && not delimiters_[data_[currentPos_]]) {
    ++currentPos_;
  }

  size_t tokenLength = currentPos_ - tokenStart;

  // Extract and convert to lowercase directly
  std::string token;
  token.reserve(tokenLength);
  for (size_t i = tokenStart; i < tokenStart + tokenLength; ++i) {
    token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(data_[i]))));
  }

  // Stem the token using Snowball
  // SN_set_current copies the token into the stem environment
  SN_set_current(stemEnv_, (int)token.size(),
                 reinterpret_cast<const unsigned char *>(token.c_str()));

  // Perform the stemming
  english_UTF_8_stem(stemEnv_);
  stemEnv_->p[stemEnv_->l] = '\0';
  return std::string(reinterpret_cast<char *>(stemEnv_->p), stemEnv_->l);
}

bool Tokenizer::hasMoreTokens() const {
  size_t tempPos = currentPos_;
  while (tempPos < size_ && delimiters_[data_[tempPos]]) {
    ++tempPos;
  }
  return tempPos < size_;
}
