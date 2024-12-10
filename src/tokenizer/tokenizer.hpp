#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cctype>
#include <string>
#include <unordered_set>

// Forward declare the snowball environment and functions
struct SN_env;
extern "C" {
SN_env *english_UTF_8_create_env();
void english_UTF_8_close_env(SN_env *);
int english_UTF_8_stem(SN_env *);
int SN_set_current(SN_env *, int, const unsigned char *);
}

#include "token.hpp"

class Tokenizer {
 public:
  Tokenizer(const char *data, size_t size);
  ~Tokenizer();

  std::string nextToken();
  bool hasMoreTokens() const;

 private:
  void skipDelimiters();

  const char *data_;
  size_t size_;
  size_t currentPos_;
  SN_env *stemEnv_;

  bool delimiters_[256]{};
};

#endif  // TOKENIZER_HPP
