#ifndef DOCUMENT_HPP
#define DOCUMENT_HPP

#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>

class Document {
    public:
    Document(int id, const char *data, size_t size, const std::shared_ptr<arrow::Buffer> &arrow_buf)
        : id(id), data(data), size(size), arrow_buf(arrow_buf) {};

    [[nodiscard]] int getId() const { return id; }

    [[nodiscard]] const char *getData() const { return data; }

    [[nodiscard]] size_t getSize() const { return size; }

    private:
    int id;
    const char *data;
    const size_t size;
    const std::shared_ptr<arrow::Buffer> &arrow_buf;
};

#endif  // DOCUMENT_HPP
