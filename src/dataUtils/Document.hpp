// Created by fts on 10/31/24.

#ifndef DOCUMENT_HPP
#define DOCUMENT_HPP
#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>

class Document {
public:
    Document(int id, const char *data, size_t size, const std::shared_ptr<arrow::Buffer> &arrowBuf);

    [[nodiscard]] uint32_t getId() const { return id; }

    [[nodiscard]] const char *getBegin() const { return beginPointer; }

    [[nodiscard]] size_t getSize() const { return size; }

private:
    uint32_t id;
    const char *beginPointer;
    const size_t size;
    const std::shared_ptr<arrow::Buffer> &arrowBuf;
};

#endif  // DOCUMENT_HPP
