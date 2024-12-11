#include "document_iterator.hpp"

#include <arrow/array.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>

DocumentIterator::DocumentIterator(const std::string &folder_path) {
  // Enqueue all Parquet files from the folder
  for (const auto &entry : fs::directory_iterator(folder_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".parquet") {
      file_queue.push(entry.path().string());
    }
  }

  loadNextFile();
}

void DocumentIterator::loadNextFile() {
  if (file_queue.empty()) {
    batch_reader.reset();
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

  PARQUET_THROW_NOT_OK(
      arrow_reader->GetRecordBatchReader({0}, &batch_reader));  // Read only column 0

  current_row_index = 0;
  total_rows_in_batch = 0;

  if (!loadNextBatch()) {
    loadNextFile();
  }
}

bool DocumentIterator::loadNextBatch() {
  if (!batch_reader) {
    return false;
  }

  PARQUET_THROW_NOT_OK(batch_reader->ReadNext(&current_batch));

  if (!current_batch || current_batch->num_rows() == 0) {
    return false;
  }

  data_array = std::dynamic_pointer_cast<arrow::BinaryArray>(current_batch->column(0));
  doc_id_array = std::dynamic_pointer_cast<arrow::UInt32Array>(current_batch->column(1));
  if (!data_array) {
    throw std::runtime_error("Column 0 is not of type BinaryArray");
  }

  total_rows_in_batch = current_batch->num_rows();
  current_row_index = 0;

  return true;
}

bool DocumentIterator::hasNext() {
  while (true) {
    if (!current_batch) {
      return false;
    }

    if (current_row_index < total_rows_in_batch) {
      return true;
    } else if (loadNextBatch()) {
      continue;
    } else {
      loadNextFile();
      if (!current_batch) {
        return false;
      }
    }
  }
}

std::shared_ptr<Document> DocumentIterator::operator*() {
  int32_t length = 0;

  const uint8_t *value = data_array->GetValue(current_row_index, &length);

  std::shared_ptr<arrow::Buffer> buffer = data_array->value_data();

  auto *data_ptr = reinterpret_cast<const char *>(value);

  uint32_t doc_id = doc_id_array->GetView(current_row_index);

  return std::make_shared<Document>(doc_id, data_ptr, length, buffer);
}

void DocumentIterator::operator++() {
  if (!hasNext()) {
    throw std::out_of_range("No more documents available");
  }
  current_row_index++;
}

void DocumentIterator::operator++(int) { ++(*this); }
