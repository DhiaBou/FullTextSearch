#ifndef DOCUMENT_ITERATOR_HPP
#define DOCUMENT_ITERATOR_HPP
#include "Document.hpp"

#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>
#include <filesystem>
#include <queue>
#include <string>
#include <memory>


namespace fs = std::filesystem;

class DocumentIterator {
public:
    explicit DocumentIterator(const std::string &folderPath);

    bool hasNext();

    std::shared_ptr<Document> operator*();

    void operator++();
    void operator++(int);

private:
    void loadNextFile();

    bool loadNextBatch();

    std::queue<std::string> fileQueue;
    std::unique_ptr<parquet::arrow::FileReader> arrowReader;
    std::shared_ptr<arrow::RecordBatchReader> batchReader;
    std::shared_ptr<arrow::RecordBatch> currentBatch;
    int64_t currentRowIndex = 0;
    int64_t totalRowsInBatch = 0;
    int documentId = 0;

    std::shared_ptr<arrow::BinaryArray> dataArray;
};

#endif  // DOCUMENT_ITERATOR_HPP
