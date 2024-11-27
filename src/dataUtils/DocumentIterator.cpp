#include "DocumentIterator.hpp"
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <arrow/memory_pool.h>

DocumentIterator::DocumentIterator(const std::string &folderPath) {
    // Enqueue all Parquet files from the folder
    for (const auto &entry: fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
            fileQueue.push(entry.path().string());
        }
    }

    loadNextFile();
}

void DocumentIterator::loadNextFile() {
    if (fileQueue.empty()) {
        batchReader.reset();
        arrowReader.reset();
        return;
    }

    std::string currentFileName = fileQueue.front();
    fileQueue.pop();


    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(
        infile,
        arrow::io::ReadableFile::Open(currentFileName, arrow::default_memory_pool()));


    PARQUET_THROW_NOT_OK(
        parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrowReader));


    PARQUET_THROW_NOT_OK(
        arrowReader->GetRecordBatchReader({0}, &batchReader)); // Read only column 0

    currentRowIndex = 0;
    totalRowsInBatch = 0;


    if (!loadNextBatch()) {
        loadNextFile();
    }
}

bool DocumentIterator::loadNextBatch() {
    if (!batchReader) {
        return false;
    }

    PARQUET_THROW_NOT_OK(batchReader->ReadNext(&currentBatch));

    if (!currentBatch || currentBatch->num_rows() == 0) {
        return false;
    }


    dataArray = std::dynamic_pointer_cast<arrow::BinaryArray>(currentBatch->column(0));
    if (!dataArray) {
        throw std::runtime_error("Column 0 is not of type BinaryArray");
    }

    totalRowsInBatch = currentBatch->num_rows();
    currentRowIndex = 0;

    return true;
}

bool DocumentIterator::hasNext() {
    while (true) {
        if (!currentBatch) {
            return false;
        }

        if (currentRowIndex < totalRowsInBatch) {
            return true;
        } else if (loadNextBatch()) {
            continue;
        } else {
            loadNextFile();
            if (!currentBatch) {
                return false;
            }
        }
    }
}
std::shared_ptr<Document> DocumentIterator::operator*() {

    int32_t length = 0;
    const uint8_t *value = dataArray->GetValue(currentRowIndex, &length);


    std::shared_ptr<arrow::Buffer> buffer = dataArray->value_data();


    const char *dataPtr = reinterpret_cast<const char *>(value);

    return std::make_shared<Document>(documentId, dataPtr, length, buffer);
}

void DocumentIterator::operator++() {
    if (!hasNext()) {
        throw std::out_of_range("No more documents available");
    }
    currentRowIndex++;
    documentId++;
}
void DocumentIterator::operator++(int) {
    ++(*this);
}
