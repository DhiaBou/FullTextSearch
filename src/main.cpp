#include <iostream>
#include <string>

#include "algorithms/inverted/inverted_index_engine.hpp"
#include "algorithms/trigram/trigram_index_engine.hpp"
#include "algorithms/vsm/vector_space_model_engine.hpp"
#include "documents/document.hpp"
#include "documents/document_iterator.hpp"
#include "fts_engine.hpp"
#include "scoring/bm25.hpp"
#include "scoring/scoring_function.hpp"
#include "scoring/tf_idf.hpp"

int main() {
  std::unique_ptr<FullTextSearchEngine> engine;

  // Specify path to the data
  std::string directory_path;
  std::cout << "Enter the absolute path to the data: ";
  std::getline(std::cin, directory_path);
  DocumentIterator it(directory_path);

  // Choose the algorithm
  std::string algorithm_choice;
  do {
    std::cout << "Select search algorithm (vsm/inverted/trigram): ";
    std::getline(std::cin, algorithm_choice);
    if (algorithm_choice == "vsm") {
      engine = std::make_unique<VectorSpaceModelEngine>();
    } else if (algorithm_choice == "inverted") {
      engine = std::make_unique<InvertedIndexEngine>();
    } else if (algorithm_choice == "trigram") {
      engine = std::make_unique<TrigramIndexEngine>();
    } else {
      std::cout << "Invalid choice!" << std::endl;
    }
  } while (engine == nullptr);

  // Build the index
  engine->indexDocuments(std::move(it));

  // Scoring
  std::unique_ptr<scoring::ScoringFunction> score_func;
  std::string scoring_choice;

  do {
    std::cout << "Select the function to score documents (bm25/tf-idf): ";
    std::getline(std::cin, scoring_choice);
    if (scoring_choice == "bm25") {
      double k1 = 1.5;
      double b = 0.75;
      score_func = std::make_unique<scoring::BM25>(engine->getDocumentCount(),
                                                   engine->getAvgDocumentLength(), k1, b);
    } else if (scoring_choice == "tf-idf") {
      score_func = std::make_unique<scoring::TfIdf>(engine->getDocumentCount());
    } else {
      std::cout << "Invalid choice!" << std::endl;
    }
  } while (score_func == nullptr);

  // Search
  std::string query;
  while (true) {
    std::cout << "Enter search query: ";
    std::getline(std::cin, query);

    auto results = engine->search(query, *score_func);

    for (const auto &doc : results) {
      std::cout << "Document ID: " << doc << std::endl;
    }
  }
}
