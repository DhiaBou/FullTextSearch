#ifndef QUERY_ITERATOR_HPP
#define QUERY_ITERATOR_HPP
//---------------------------------------------------------------------------
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
//---------------------------------------------------------------------------
namespace fs = std::filesystem;
//---------------------------------------------------------------------------
namespace queries {
//---------------------------------------------------------------------------
struct Query {
  std::string id;
  std::string content;
};
//---------------------------------------------------------------------------
class QueryIterator {
 public:
  enum class Type : unsigned { File, CommandLine };
  virtual ~QueryIterator() = default;
  virtual Query next() = 0;
  virtual bool hasNext() = 0;
  virtual Type getType() const = 0;
};
//---------------------------------------------------------------------------
class FileIterator : public QueryIterator {
 public:
  explicit FileIterator(const std::string& path);
  ~FileIterator() override = default;
  bool hasNext() override;
  Query next() override;
  Type getType() const override;

 private:
  size_t current;
  std::vector<Query> queries;
};
//---------------------------------------------------------------------------
class CommandLineIterator : public QueryIterator {
 public:
  CommandLineIterator();
  ~CommandLineIterator() override = default;
  Query next() override;
  bool hasNext() override;
  Type getType() const override;

 private:
  size_t count;
  Query query;
};
//---------------------------------------------------------------------------
}  // namespace queries
//---------------------------------------------------------------------------
#endif  // TRIGRAM_ITERATOR_HPP
