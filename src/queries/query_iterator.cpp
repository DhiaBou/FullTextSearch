#include "query_iterator.hpp"
//---------------------------------------------------------------------------
#include "utils.hpp"
//---------------------------------------------------------------------------
namespace queries {
//---------------------------------------------------------------------------
FileIterator::FileIterator(const std::string& path) : current(0) {
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.path().extension() == ".txt") {
      // Only .txt files contain queries

      query_utils::FileReader query_input(entry.path().c_str());
      queries.push_back(
          {{entry.path().stem().string()}, {query_input.begin(), query_input.getSize()}});
    }
  }
}
//---------------------------------------------------------------------------
bool FileIterator::hasNext() { return current < queries.size(); }
//---------------------------------------------------------------------------
Query FileIterator::next() { return queries[current++]; }
//---------------------------------------------------------------------------
QueryIterator::Type FileIterator::getType() const { return Type::File; }
//---------------------------------------------------------------------------
CommandLineIterator::CommandLineIterator() : count(0) {}
//---------------------------------------------------------------------------
bool CommandLineIterator::hasNext() {
  std::cout << "Enter query (!q to quit): ";
  query.id = std::to_string(count);
  std::getline(std::cin, query.content);

  if (query.content == "!q") {
    return false;
  }

  ++count;
  return true;
}
//---------------------------------------------------------------------------
Query CommandLineIterator::next() { return query; }
//---------------------------------------------------------------------------
QueryIterator::Type CommandLineIterator::getType() const { return Type::CommandLine; }
//---------------------------------------------------------------------------
}  // namespace queries