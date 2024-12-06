#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cctype>
#include <string>
#include <unordered_set>

#include "token.hpp"

class Tokenizer {
 public:
  Tokenizer(const char *data, size_t size);

  Token nextToken();

  bool hasMoreTokens() const;

 private:
  void skipDelimiters();

  const char *data_;
  size_t size_;
  size_t currentPos_;

  static const std::unordered_set<char> delimiters_;
};

#endif  // TOKENIZER_HPP
