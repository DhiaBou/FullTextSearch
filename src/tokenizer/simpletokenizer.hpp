#ifndef SIMPLETOKENIZER_HPP
#define SIMPLETOKENIZER_HPP

#include <cctype>
#include <cstring>
#include <string>

#include "ITokenizer.hpp"
namespace tokenizer {
class SimpleTokenizer : public ITokenizer {
 public:
  SimpleTokenizer(const char* data, size_t size);
  ~SimpleTokenizer() override = default;

  std::string nextToken(bool skip_stop_words) override;

 private:
  void skipDelimiters();

  const char* data_;
  size_t size_;
  size_t currentPos_;
};
}  // namespace tokenizer
#endif  // SIMPLETOKENIZER_HPP
