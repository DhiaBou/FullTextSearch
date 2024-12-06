#include "tokenizer.hpp"

Tokenizer::Tokenizer(const char *data, size_t size) : data_(data), size_(size), currentPos_(0) {}

const std::unordered_set<char> Tokenizer::delimiters_ = {
    ' ',  '\t', '\n', '\r', '.', ',', ';', ':', '!', '(', ')', '[', ']', '{', '}', '<', '>',  '?',
    '\"', '\'', '`',  '~',  '@', '#', '$', '%', '^', '&', '*', '-', '_', '=', '+', '|', '\\', '/',
    // Add any additional delimiters as needed
};

void Tokenizer::skipDelimiters() {
  while (currentPos_ < size_ && delimiters_.find(data_[currentPos_]) != delimiters_.end()) {
    ++currentPos_;
  }
}

Token Tokenizer::nextToken() {
  skipDelimiters();

  if (currentPos_ >= size_) {
    return {};
  }

  // Find the beginning of the token by advancing until a delimiter is not found anymore
  while (currentPos_ < size_ && delimiters_.contains(data_[currentPos_])) {
    ++currentPos_;
  }

  size_t tokenStart = currentPos_;

  // Find the end of the token by advancing until a delimiter is found
  while (currentPos_ < size_ && !delimiters_.contains(data_[currentPos_])) {
    ++currentPos_;
  }

  size_t tokenLength = currentPos_ - tokenStart;

  // Extract the token and convert it to lowercase
  std::string token;
  token.reserve(tokenLength);
  for (size_t i = tokenStart; i < currentPos_; ++i) {
    token += static_cast<char>(std::tolower(static_cast<unsigned char>(data_[i])));
  }

  return token;
}

bool Tokenizer::hasMoreTokens() const {
  size_t tempPos = currentPos_;
  while (tempPos < size_ && delimiters_.find(data_[tempPos]) != delimiters_.end()) {
    ++tempPos;
  }
  return tempPos < size_;
}
