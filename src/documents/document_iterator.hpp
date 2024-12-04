#ifndef DOCUMENT_ITERATOR_HPP
#define DOCUMENT_ITERATOR_HPP

#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>
#include <filesystem>
#include <queue>
#include <string>
#include <memory>
#include "document.hpp"

namespace fs = std::filesystem;

class DocumentIterator {
public:
    explicit DocumentIterator(const std::string &folder_path);

    bool hasNext();

    std::shared_ptr<Document> operator*();

    void operator++();
    void operator++(int);

private:
    void loadNextFile();

    bool loadNextBatch();

    std::queue<std::string> file_queue;
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    std::shared_ptr<arrow::RecordBatch> current_batch;
    int64_t current_row_index = 0;
    int64_t total_rows_in_batch = 0;
    int doc_id = 0;

    std::shared_ptr<arrow::BinaryArray> data_array;
};

#endif  // DOCUMENT_ITERATOR_HPP
