#include "document_iterator.hpp"

#include <arrow/array.h>
#include <arrow/memory_pool.h>
#include <arrow/table.h>

#include <iostream>
#include <thread>

DocumentIterator::DocumentIterator(const std::string &folder_path, uint32_t batch_size)
    : num_row_groups(0), row_group_index(0), batch_size(batch_size), row_batch_index(0) {
  // Enqueue all Parquet files from the folder
  for (const auto &entry : fs::directory_iterator(folder_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
      file_queue.push(entry.path().string());
    }
  }
}

void DocumentIterator::loadNextFile() {
  if (file_queue.empty()) {
    arrow_reader.reset();
    return;
  }

  std::string current_file_name = file_queue.front();
  file_queue.pop();

  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(
      infile, arrow::io::ReadableFile::Open(current_file_name, arrow::default_memory_pool()));

  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrow_reader));

  num_row_groups = static_cast<uint32_t>(arrow_reader->num_row_groups());
  row_group_index = 0;

  loadNextRowGroup();
}

bool DocumentIterator::loadNextRowGroup() {
  if (row_group_index == num_row_groups) return false;

  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(arrow_reader->ReadRowGroup(row_group_index, &table));
  PARQUET_ASSIGN_OR_THROW(table, table->CombineChunks());

  std::shared_ptr<arrow::ChunkedArray> data_column = table->column(0);
  std::shared_ptr<arrow::ChunkedArray> id_column = table->column(1);

  content_array = std::dynamic_pointer_cast<arrow::BinaryArray>(data_column->chunk(0));
  doc_id_array = std::dynamic_pointer_cast<arrow::UInt32Array>(id_column->chunk(0));

  if (!content_array) {
    throw std::runtime_error("Column 0 is not of type BinaryArray");
  }

  ++row_group_index;
  row_batch_index = 0;

  return true;
}

bool DocumentIterator::hasNext() { return true; }

std::shared_ptr<Document> DocumentIterator::operator*() {
  int32_t length = 0;

  const uint8_t *value = content_array->GetValue(0, &length);

  std::shared_ptr<arrow::Buffer> buffer = content_array->value_data();

  auto *data_ptr = reinterpret_cast<const char *>(value);

  uint32_t doc_id = doc_id_array->GetView(0);

  return std::make_shared<Document>(doc_id, data_ptr, length, buffer);
}

void DocumentIterator::operator++() {
  if (!hasNext()) {
    throw std::out_of_range("No more documents available");
  }
}

void DocumentIterator::operator++(int) { ++(*this); }

void DocumentIterator::readBatch(size_t start, size_t end, std::vector<Document> &docs) {
  for (size_t current = start; current < end; ++current) {
    // Read row
    int32_t length = 0;

    const uint8_t *value = content_array->GetValue(current, &length);
    std::shared_ptr<arrow::Buffer> buffer = content_array->value_data();
    auto *data_ptr = reinterpret_cast<const char *>(value);

    uint32_t doc_id = doc_id_array->GetView(current);

    // Insert into document vector
    docs[current & (batch_size - 1)] = Document(doc_id, data_ptr, length, buffer);
  }
}

std::vector<Document> DocumentIterator::next() {
  std::unique_lock lck(global_lock);

  // current file exhausted
  if (row_group_index == num_row_groups) {
    loadNextFile();
  }

  // current row group exhausted
  if (row_batch_index * batch_size >= content_array->length()) {
    if (!loadNextRowGroup()) {
      return {};
    }
  }

  // read the next batch

  // for the last batch, there might not be batch_size elements left
  uint32_t real_batch_size = std::min(
      batch_size, static_cast<uint32_t>(content_array->length() - batch_size * row_batch_index));

  std::vector<Document> docs(real_batch_size);
  readBatch(batch_size * row_batch_index,
            std::min(static_cast<size_t>(batch_size * (row_batch_index + 1)),
                     static_cast<size_t>(content_array->length())),
            docs);

  ++row_batch_index;

  return docs;
}
