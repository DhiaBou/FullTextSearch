//
// Created by dhia on 12/11/24.
//

#ifndef TOKENIZER_RULES_HPP
#define TOKENIZER_RULES_HPP
#include <string>
#include <unordered_set>
namespace tokenizer {
static const char DELIM_CHARS[] = " \t\n\r.,;:!()[]{}<>?\"'~@#$%^&*-_=+|\\";
static const std::unordered_set<std::string> STOP_WORDS = {
    "i",       "me",      "my",      "myself",   "we",         "our",    "ours",    "ourselves",
    "you",     "your",    "yours",   "yourself", "yourselves", "he",     "him",     "his",
    "himself", "she",     "her",     "hers",     "herself",    "it",     "its",     "itself",
    "they",    "them",    "their",   "theirs",   "themselves", "what",   "which",   "who",
    "whom",    "this",    "that",    "these",    "those",      "am",     "is",      "are",
    "was",     "were",    "be",      "been",     "being",      "have",   "has",     "had",
    "having",  "do",      "does",    "did",      "doing",      "a",      "an",      "the",
    "and",     "but",     "if",      "or",       "because",    "as",     "until",   "while",
    "of",      "at",      "by",      "for",      "with",       "about",  "against", "between",
    "into",    "through", "during",  "before",   "after",      "above",  "below",   "to",
    "from",    "up",      "down",    "in",       "out",        "on",     "off",     "over",
    "under",   "again",   "further", "then",     "once",       "here",   "there",   "when",
    "where",   "why",     "how",     "all",      "any",        "both",   "each",    "few",
    "more",    "most",    "other",   "some",     "such",       "no",     "nor",     "not",
    "only",    "own",     "same",    "so",       "than",       "too",    "very",    "s",
    "t",       "can",     "will",    "just",     "don",        "should", "now"};
static bool DELIMS[256] = {false};

static const auto init_delims = [] {
  for (const char* p = DELIM_CHARS; *p != '\0'; ++p) {
    DELIMS[static_cast<unsigned char>(*p)] = true;
  }
  return 0;
}();

static constexpr bool isDelimiter(const char c) { return DELIMS[static_cast<unsigned char>(c)]; }

static constexpr bool isStopWord(const std::string& word) { return STOP_WORDS.contains(word); }

}  // namespace tokenizer
#endif  // TOKENIZER_RULES_HPP
