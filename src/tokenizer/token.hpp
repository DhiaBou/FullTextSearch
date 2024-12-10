//
// Created by dhia on 12/4/24.
//

#ifndef TOKEN_HPP
#define TOKEN_HPP

#include <functional>
#include <string>

struct Token {
  Token() = default;

  Token(std::string token)
      : token_(std::move(token))  // NOLINT implicit is OK
  {}

  [[nodiscard]] const std::string &getString() const { return token_; }

  bool operator==(const Token &other) const { return token_ == other.token_; }

  bool operator!=(const Token &other) const { return !(*this == other); }

  bool empty() const { return token_.empty(); }

 private:
  std::string token_;
};

#endif  // TOKEN_HPP
