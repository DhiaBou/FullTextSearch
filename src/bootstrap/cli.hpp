#ifndef CLI_HPP
#define CLI_HPP
//---------------------------------------------------------------------------
#include <cxxopts.hpp>
#include <string>
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
struct FTSOptions {
  std::string data_path;
  std::string algorithm;
  std::string scoring;
  uint32_t num_results;
  std::string queries_path;
  std::string output_prefix;
  bool benchmarking_mode;
};
//---------------------------------------------------------------------------
FTSOptions parseCommandLine(int argc, char** argv);
//---------------------------------------------------------------------------
}  // namespace bootstrap
//---------------------------------------------------------------------------
#endif  // CLI_HPP