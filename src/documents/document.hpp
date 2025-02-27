#ifndef DOCUMENT_HPP
#define DOCUMENT_HPP

#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>

#include <cstdint>

/**
 * Encapsulates the data and metadata of a document that is indexed
 * and searched by the full-text search engines.
 */
class Document {
 public:
  /// Default constructor.
  Document() : id(0), data(nullptr), size(0), arrow_buf(nullptr) {}
  /// Constructor.
  Document(uint32_t id, const char *data, size_t size, std::shared_ptr<arrow::Buffer> arrow_buf)
      : id(id), data(data), size(size), arrow_buf(std::move(arrow_buf)) {};
  /// Move constructor.
  Document(Document &&other) noexcept
      : id(other.id), data(other.data), size(other.size), arrow_buf(std::move(other.arrow_buf)) {}
  /// Move assigment.
  Document &operator=(Document &&other) noexcept {
    if (this != &other) {
      id = other.id;
      data = other.data;
      size = other.size;
      arrow_buf = std::move(other.arrow_buf);
    }
    return *this;
  }
  /// Copy Constructor.
  Document(const Document &) = delete;
  /// Copy assigment.
  Document &operator=(const Document &) = delete;

  /// Get the document's ID.
  [[nodiscard]] uint32_t getId() const { return id; }
  /// Get a pointer to the document's data.
  [[nodiscard]] const char *getData() const { return data; }
  /// Get the document's size in bytes.
  [[nodiscard]] size_t getSize() const { return size; }

 private:
  /// The document's ID.
  uint32_t id;
  /// A pointer to the document's data.
  const char *data;
  /// The document's size in bytes.
  size_t size;
  /// A pointer to the document's arrow buffer.
  /// Required to keep the data pointer valid.
  std::shared_ptr<arrow::Buffer> arrow_buf;
};

#endif  // DOCUMENT_HPP
