#ifndef DOCUMENT_ITERATOR_HPP
#define DOCUMENT_ITERATOR_HPP

#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>
#include <filesystem>
#include <queue>
#include <string>
#include <memory>
#include <cstdint>
#include "document.hpp"

namespace fs = std::filesystem;

/**
 * Iterator for traversing documents stored in Parquet files.
 */
class DocumentIterator {
public:
    /// Constructor.
    explicit DocumentIterator(const std::string &folder_path);
    /// Prefix increment operator.
    void operator++();
    /// Postfix increment operator.
    void operator++(int);
    /// Dereference operator.
    std::shared_ptr<Document> operator*();
    /// Whether there is another document.
    bool hasNext();

private:
    /// Load the next file in the given directory.
    void loadNextFile();
    /// Load the next batch of rows.
    bool loadNextBatch();

    /// A queue of parquet files contained in the specified directory.
    std::queue<std::string> file_queue;
    /// An abstraction used to read parquet files into arrow batches.
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    /// An abstraction used to read rows into arrow batches.
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    /// The current batch the iterator is working on.
    std::shared_ptr<arrow::RecordBatch> current_batch;
    /// The current batch's raw data.
    std::shared_ptr<arrow::BinaryArray> data_array;
    /// The current row of the iterator's current batch.
    int64_t current_row_index = 0;
    /// The total number of rows in the iterator's current batch.
    int64_t total_rows_in_batch = 0;
    /// The current document's identifier that is incremented with each row.
    uint32_t doc_id = 0;
};

#endif  // DOCUMENT_ITERATOR_HPP
