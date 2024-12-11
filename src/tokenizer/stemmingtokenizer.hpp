#ifndef STEMMINGTOKENIZER_HPP
#define STEMMINGTOKENIZER_HPP

#include <cctype>
#include <string>

#include "ITokenizer.hpp"

struct SN_env;
extern "C" {
SN_env *english_UTF_8_create_env();
void english_UTF_8_close_env(SN_env *);
int english_UTF_8_stem(SN_env *);
int SN_set_current(SN_env *, int, const unsigned char *);
}
namespace tokenizer {

class StemmingTokenizer : public ITokenizer {
 public:
  StemmingTokenizer(const char *data, size_t size);
  ~StemmingTokenizer() override;

  std::string nextToken(bool skip_stop_words) override;

 private:
  void skipDelimiters();

  const char *data_;
  size_t size_;
  size_t currentPos_;
  SN_env *stemEnv_;
};
}  // namespace tokenizer
#endif  // STEMMINGTOKENIZER_HPP
