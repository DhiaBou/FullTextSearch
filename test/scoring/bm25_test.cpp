#include "scoring/bm25.hpp"

#include <gtest/gtest.h>

TEST(BM25Test, ScoreCalculation) {
  scoring::BM25 bm25(4001, 1224.43, 1.5, 0.75);
  scoring::DocStats doc_stats = {1000};
  scoring::WordStats word_stats = {10, 234};

  // calculated using wolframalpha
  double expected_score = 6.28021;
  double actual_score = bm25.score(doc_stats, word_stats);

  EXPECT_NEAR(actual_score, expected_score, 1e-5);
}

TEST(BM25Test, ZeroFrequency) {
  scoring::BM25 bm25(4001, 1224.43, 1.5, 0.75);
  scoring::DocStats doc_stats = {1000};
  scoring::WordStats word_stats = {0, 234};

  double expected_score = 0.0;
  double actual_score = bm25.score(doc_stats, word_stats);

  EXPECT_NEAR(actual_score, expected_score, 1e-5);
}

TEST(BM25Test, HighConstants) {
  scoring::BM25 bm25(564000, 3389721.837, 1.5, 0.75);
  scoring::DocStats doc_stats = {12000000};
  scoring::WordStats word_stats = {1, 133987};

  // calculated using wolframalpha
  double expected_score = 0.670684;
  double actual_score = bm25.score(doc_stats, word_stats);

  EXPECT_NEAR(actual_score, expected_score, 1e-5);
}