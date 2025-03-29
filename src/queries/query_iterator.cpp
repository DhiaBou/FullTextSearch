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

      utils::FileReader query_input(entry.path().c_str());

      auto word_begin = query_input.begin();
      auto word_end = query_input.begin();
      auto end = query_input.end();
      while (word_begin != end) {
        while (*word_end != '\n' && word_end != end) ++word_end;
        queries.push_back({{entry.path().stem().string()},
                           {word_begin, static_cast<size_t>(word_end - word_begin)}});
        if (word_end == end) break;
        ++word_end;
        word_begin = word_end;
      }
    }
  }  // namespace queries
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