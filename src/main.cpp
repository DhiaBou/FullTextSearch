#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "algorithms/inverted/inverted_index_engine.hpp"
#include "algorithms/trigram/trigram_index_engine.hpp"
#include "algorithms/vsm/vector_space_model_engine.hpp"
#include "documents/document_iterator.hpp"
#include "fts_engine.hpp"
#include "scoring/bm25.hpp"
#include "scoring/scoring_function.hpp"
#include "scoring/tf_idf.hpp"

int main(int argc, char** argv) {
  cxxopts::Options options("Fulltext search",
                           "Multiple implementations of different fulltext search algorithms");

  options.add_options()("d,data", "Path to the directory containing all data",
                        cxxopts::value<std::string>())(
      "a,algorithm", "Algorithm (inverted/vsm/trigram)", cxxopts::value<std::string>())(
      "s,scoring", "Scoring (tf-idf,bm25)", cxxopts::value<std::string>())("h,help", "Print usage");
  auto result = options.parse(argc, argv);

  if (result.count("data") == 0 || result.count("algorithm") == 0 || result.count("scoring") == 0) {
    std::cout << options.help() << std::endl;
    return 1;
  }

  auto directory_path = result["data"].as<std::string>();
  DocumentIterator it(directory_path);

  auto algorithm_choice = result["algorithm"].as<std::string>();
  std::unique_ptr<FullTextSearchEngine> engine;
  if (algorithm_choice == "vsm") {
    engine = std::make_unique<VectorSpaceModelEngine>();
  } else if (algorithm_choice == "inverted") {
    engine = std::make_unique<InvertedIndexEngine>();
  } else if (algorithm_choice == "trigram") {
    engine = std::make_unique<TrigramIndexEngine>();
  } else {
    std::cout << options.help() << std::endl;
    return 1;
  }

  // Build the index
  engine->indexDocuments(std::move(it));

  // Scoring
  std::unique_ptr<scoring::ScoringFunction> score_func;
  auto scoring_choice = result["scoring"].as<std::string>();
  if (scoring_choice == "bm25") {
    double k1 = 1.5;
    double b = 0.75;
    score_func = std::make_unique<scoring::BM25>(engine->getDocumentCount(),
                                                 engine->getAvgDocumentLength(), k1, b);
  } else if (scoring_choice == "tf-idf") {
    score_func = std::make_unique<scoring::TfIdf>(engine->getDocumentCount());
  } else {
    std::cout << options.help() << std::endl;
    return 1;
  }

  std::string query;
  while (true) {
    std::cout << "Enter query (!q to quit): ";
    std::getline(std::cin, query);

    if (query == "!q") {
      break;
    }

    auto results = engine->search(query, *score_func);

    for (const auto& doc : results) {
      std::cout << doc << std::endl;
    }
  }

  return 0;
}
