//
// Created by dhia on 12/11/24.
//

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "tokenizer/simpletokenizer.hpp"
#include "tokenizer/stemmingtokenizer.hpp"

namespace tokenizer {

// Test for basic tokenization and stemming
TEST(StemmingTokenizerTest, BasicTokenizationAndStemming) {
  const std::string input = ",.!?;:\nrunning[] {}<>\\, jumped,!?;:\"() quickly.! !  ";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  // Expected stemmed tokens
  std::vector<std::string> expectedTokens = {"run", "jump", "quick"};

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);  // No stop word filtering
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_EQ(tokens, expectedTokens);
}

// Test for handling stop words
TEST(StemmingTokenizerTest, SkipStopWords) {
  const std::string input = "The quick brown fox jumps over the lazy dog";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  // Expected tokens after stop word filtering
  std::vector<std::string> expectedTokens = {"quick", "brown", "fox", "jump", "lazi", "dog"};

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(true);  // Enable stop word filtering
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_EQ(tokens, expectedTokens);
}

// Test for case insensitivity
TEST(StemmingTokenizerTest, CaseInsensitiveTokenization) {
  const std::string input = "Running RUNNING runNiNg running";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  // Expected stemmed tokens
  std::vector<std::string> expectedTokens = {"run", "run", "run", "run"};

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);  // No stop word filtering
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_EQ(tokens, expectedTokens);
}
// Test for empty input
TEST(StemmingTokenizerTest, EmptyInput) {
  const std::string input = "";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_TRUE(tokens.empty());
}

// Test for handling all delimiters
TEST(StemmingTokenizerTest, AllDelimiters) {
  const std::string input = " \t\n\r.,;:!()[]{}<>?\"'~@#$%^&*-_=+|\\";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_TRUE(tokens.empty());
}

// Test for input with only stop words
TEST(StemmingTokenizerTest, OnlyStopWords) {
  const std::string input = "the is and a an of over";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(true);  // Enable stop word filtering
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_TRUE(tokens.empty());
}

// Test for mixed delimiters and tokens
TEST(StemmingTokenizerTest, MixedDelimitersAndTokens) {
  const std::string input = "token1, token2.token3!token4?token5";
  StemmingTokenizer tokenizer(input.c_str(), input.size());

  std::vector<std::string> expectedTokens = {"token1", "token2", "token3", "token4", "token5"};
  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_EQ(tokens, expectedTokens);
}

// Test for Unicode characters
TEST(StemmingTokenizerTest, UnicodeCharacters) {
  const std::string input = "résumés café naïve jalapeño";
  SimpleTokenizer tokenizer(input.c_str(), input.size());

  // Note: Stemming rules for these words might vary; here we expect no stemming due to Unicode
  // handling.
  std::vector<std::string> expectedTokens = {"résumés", "café", "naïve", "jalapeño"};
  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_EQ(tokens, expectedTokens);
}

// Test for edge case with extremely long token
TEST(StemmingTokenizerTest, ExtremelyLongToken) {
  std::string longToken(10000, 'a');  // A very long string of 'a's
  StemmingTokenizer tokenizer(longToken.c_str(), longToken.size());

  std::vector<std::string> tokens;
  while (true) {
    std::string token = tokenizer.nextToken(false);
    if (token.empty()) break;
    tokens.push_back(token);
  }

  EXPECT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], longToken);
}

}  // namespace tokenizer
