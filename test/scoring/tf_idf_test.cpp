#include "scoring/tf_idf.hpp"

#include <gtest/gtest.h>

TEST(TfIdfTest, ScoreCalculation) {
  scoring::TfIdf tfidf(4001);
  scoring::DocStats doc_stats = {1000};
  scoring::WordStats word_stats = {10, 234};

  // calculated using wolframalpha
  double expected_score = 0.0283709393;
  double actual_score = tfidf.score(doc_stats, word_stats);

  EXPECT_NEAR(actual_score, expected_score, 1e-5);
}

TEST(TfIdfTest, ZeroFrequency) {
  scoring::TfIdf tfidf(4001);
  scoring::DocStats doc_stats = {1000};
  scoring::WordStats word_stats = {0, 234};

  double expected_score = 0.0;
  double actual_score = tfidf.score(doc_stats, word_stats);

  EXPECT_NEAR(actual_score, expected_score, 1e-5);
}

TEST(TfIdfTest, HighConstants) {
  scoring::TfIdf tfidf(564000);
  scoring::DocStats doc_stats = {12000000};
  scoring::WordStats word_stats = {1, 133987};

  // calculated using wolframalpha
  double expected_score = 0.000000119775;
  double actual_score = tfidf.score(doc_stats, word_stats);

  EXPECT_NEAR(actual_score, expected_score, 1e-5);
}