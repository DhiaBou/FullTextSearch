#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
//---------------------------------------------------------------------------
#include "algorithms/inverted/inverted_index_engine.hpp"
#include "algorithms/trigram/trigram_index_engine.hpp"
#include "algorithms/vsm/vector_space_model_engine.hpp"
#include "bootstrap/cli.hpp"
#include "documents/document_iterator.hpp"
#include "fts_engine.hpp"
#include "queries/query_iterator.hpp"
#include "scoring/bm25.hpp"
#include "scoring/scoring_function.hpp"
#include "scoring/tf_idf.hpp"
//---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Parse input arguments
  auto options = bootstrap::parseCommandLine(argc, argv);

  // Prepare iterating over the documents
  DocumentIterator it(options.data_path);

  // Decide for a FTS-Index-Engine
  auto algorithm_choice = std::move(options.algorithm);
  std::unique_ptr<FullTextSearchEngine> engine;
  if (algorithm_choice == "vsm") {
    engine = std::make_unique<VectorSpaceModelEngine>();
  } else if (algorithm_choice == "inverted") {
    engine = std::make_unique<InvertedIndexEngine>();
  } else if (algorithm_choice == "trigram") {
    engine = std::make_unique<TrigramIndexEngine>();
  } else {
    throw std::invalid_argument("Invalid algorithm choice!");
  }

  // Define the scoring function used to score documents
  std::unique_ptr<scoring::ScoringFunction> score_func;
  auto scoring_choice = std::move(options.scoring);
  if (scoring_choice == "bm25") {
    score_func =
        std::make_unique<scoring::BM25>(engine->getDocumentCount(), engine->getAvgDocumentLength());
  } else if (scoring_choice == "tf-idf") {
    score_func = std::make_unique<scoring::TfIdf>(engine->getDocumentCount());
  } else {
    throw std::invalid_argument("Invalid scoring choice!");
  }

  // Build the FTS-Index
  engine->indexDocuments(std::move(it));

  // Execute queries on the FTS-Index
  std::unique_ptr<queries::QueryIterator> query_engine;
  if (options.queries_path.size() > 0) {
    // Read and write queries from and to files
    query_engine = std::make_unique<queries::FileIterator>(options.queries_path);
  } else {
    // Read and write queries from and to console
    query_engine = std::make_unique<queries::CommandLineIterator>();
  }

  while (query_engine->hasNext()) {
    queries::Query query = query_engine->next();
    auto results = engine->search(query.content, *score_func, options.num_results);

    if (query_engine->getType() == queries::QueryIterator::Type::File) {
      fs::path output_path = fs::path(options.queries_path) / (query.id + "_result.tbl");
      std::ofstream file_output(output_path);
      for (const auto& [doc_id, score] : results) {
        file_output << doc_id << "|" << score << "|" << std::endl;
      }
    } else {
      for (const auto& [doc_id, score] : results) {
        std::cout << doc_id << "|" << score << "|" << std::endl;
      }
    }
  }

  return 0;
}
