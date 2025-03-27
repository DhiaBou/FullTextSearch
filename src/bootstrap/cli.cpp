#include <iostream>
//---------------------------------------------------------------------------
#include "cli.hpp"
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
FTSOptions parseCommandLine(int argc, char** argv) {
  cxxopts::Options options("Fulltext search",
                           "Implementations of different fulltext search algorithms");

  // clang-format off
  options.add_options()
    ("d,data", "Path to the directory containing all data", cxxopts::value<std::string>())
    ("a,algorithm", "Algorithm (inverted/vsm/trigram)", cxxopts::value<std::string>())
    ("s,scoring", "Scoring (tf-idf,bm25)", cxxopts::value<std::string>())
    ("b,benchmarking-mode", "Run in benchmark mode, no queries", cxxopts::value<bool>()->default_value("false"))
    ("n,num_results", "Number of results displayed per query", cxxopts::value<uint32_t>()->default_value("10"))
    (
      "q,queries",
      "Optional: Specifies the path to a directory containing .txt files. Each file represents a single query. "\
      "Also, it represents the output directory of the query results.", cxxopts::value<std::string>()
    )
    ("h,help", "Print usage");
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result.count("data") == 0 || result.count("algorithm") == 0 || result.count("scoring") == 0) {
    std::cout << options.help() << std::endl;
    exit(1);
  }

  FTSOptions opts;
  opts.data_path = result["data"].as<std::string>();
  opts.algorithm = result["algorithm"].as<std::string>();
  opts.scoring = result["scoring"].as<std::string>();
  opts.num_results = result["num_results"].as<uint32_t>();
  opts.benchmarking_mode = result["benchmarking-mode"].as<bool>();
  if (result.count("queries")) {
    opts.queries_path = result["queries"].as<std::string>();
  }

  return opts;
}
//---------------------------------------------------------------------------
}  // namespace bootstrap