#ifndef ITOKENIZER_HPP
#define ITOKENIZER_HPP

#include <string>
namespace tokenizer {
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // Returns the next token. If no more tokens are available, returns an empty string.
  virtual std::string nextToken(bool skip_stop_words) = 0;
};
}  // namespace tokenizer
#endif  // ITOKENIZER_HPP
