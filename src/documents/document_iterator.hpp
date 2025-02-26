#ifndef DOCUMENT_ITERATOR_HPP
#define DOCUMENT_ITERATOR_HPP

#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <queue>
#include <string>

#include "document.hpp"

namespace fs = std::filesystem;

/**
 * Iterator for traversing documents stored in Parquet files.
 */
class DocumentIterator {
 public:
  /// Constructor.
  explicit DocumentIterator(const std::string &folder_path, uint32_t batch_size = 128);

  /// @brief Produces the next batch of documents.
  /// @return The produced batch of documents.
  /// Empty if there are no documents left.
  std::vector<Document> next();

 private:
  /// Load the next file in the given directory.
  void loadNextFile();
  /// Load the next row group.
  /// @return False if there is no row group left.
  bool loadNextRowGroup();
  /// Read raw data within provided borders into the document vector.
  void readBatch(size_t start, size_t num_rows, std::vector<Document> &docs);

  /// A queue of parquet files contained in the specified directory.
  std::queue<std::string> file_queue;
  /// An abstraction used to read parquet files into arrow batches.
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  /// The current batch's raw content data.
  std::shared_ptr<arrow::BinaryArray> content_array;
  /// The current batch's raw document ID data.
  std::shared_ptr<arrow::UInt32Array> doc_id_array;

  /// The number of row groups in current file.
  uint32_t num_row_groups;
  /// The current row group.
  uint32_t row_group_index;
  /// The number of documents in a single batch.
  uint32_t batch_size;
  /// The index of the current row batch.
  uint32_t row_batch_index;

  /// A global lock on the next function to sychronize multiple threads.
  std::mutex global_lock;
};

#endif  // DOCUMENT_ITERATOR_HPP
