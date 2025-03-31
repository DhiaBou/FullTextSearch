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
#include "fts_engine.hpp"
#include "queries/query_iterator.hpp"
#include "scoring/bm25.hpp"
#include "scoring/scoring_function.hpp"
#include "scoring/tf_idf.hpp"
//---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Parse input arguments
  auto options = bootstrap::parseCommandLine(argc, argv);

  // Decide for a FTS-Index-Engine
  auto& algorithm_choice = options.algorithm;
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

  // Build the FTS-Index
  std::cout << "Build: ";
  {
    auto start = std::chrono::high_resolution_clock::now();
    engine->indexDocuments(options.data_path);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << std::endl;
  }

  // Determine the memory footprint
  const auto& allocated_footprint = engine->footprint_capacity();
  std::cout << "Allocated memory footprint: " << allocated_footprint << std::endl;
  const auto& real_footprint = engine->footprint_size();
  std::cout << "Real memory footprint: " << real_footprint << std::endl;

  if (options.benchmarking_mode) {
    return 0;
  }

  // Define the scoring function used to score documents
  std::unique_ptr<scoring::ScoringFunction> score_func;
  auto& scoring_choice = options.scoring;
  if (scoring_choice == "bm25") {
    score_func =
        std::make_unique<scoring::BM25>(engine->getDocumentCount(), engine->getAvgDocumentLength());
  } else if (scoring_choice == "tf-idf") {
    score_func = std::make_unique<scoring::TfIdf>(engine->getDocumentCount());
  } else {
    throw std::invalid_argument("Invalid scoring choice!");
  }

  // Execute queries on the FTS-Index
  std::unique_ptr<queries::QueryIterator> query_engine;
  if (options.queries_path.size() > 0) {
    // Read and write queries from and to files
    query_engine = std::make_unique<queries::FileIterator>(options.queries_path);
  } else {
    // Read and write queries from and to console
    query_engine = std::make_unique<queries::CommandLineIterator>();
  }

  // Read queries
  if (query_engine->getType() == queries::QueryIterator::Type::File) {
    fs::path output_path = fs::path(options.queries_path) / (options.output_prefix + "_result.tbl");
    std::ofstream file_output(output_path);
    while (query_engine->hasNext()) {
      queries::Query query = query_engine->next();

      // Run query
      std::vector<std::pair<DocumentID, double>> results;
      std::cout << query.content << ": ";
      {
        auto start = std::chrono::high_resolution_clock::now();
        results = engine->search(query.content, *score_func, options.num_results);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                  << std::endl;
      }
      for (const auto& [doc_id, score] : results) {
        file_output << query.content << "|" << doc_id << "|" << score << "|" << std::endl;
      }
    }
  } else {
    while (query_engine->hasNext()) {
      queries::Query query = query_engine->next();

      // Run query
      std::vector<std::pair<DocumentID, double>> results;
      std::cout << query.content << ": ";
      {
        auto start = std::chrono::high_resolution_clock::now();
        results = engine->search(query.content, *score_func, options.num_results);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                  << std::endl;
      }
      for (const auto& [doc_id, score] : results) {
        std::cout << doc_id << "|" << score << "|" << std::endl;
      }
    }
  }

  return 0;
}
