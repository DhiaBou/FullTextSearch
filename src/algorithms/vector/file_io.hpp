
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

class MmapedFileWriter {
 public:
  MmapedFileWriter(const char *path, size_t initial_max_size) : current_max_size(initial_max_size) {
    // open the file
    this->file_descriptor = open(path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (file_descriptor < 0) exit(1);

    // set file size of the mapped file to intermediate_size
    if (ftruncate(file_descriptor, initial_max_size)) {
      std::cout << "setting the intermediate file size did not work.\n";
      close(file_descriptor);
      exit(1);
    }

    // map the specified file into memory
    data = static_cast<char *>(
        mmap(nullptr, initial_max_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor, 0));
  };

  ~MmapedFileWriter() {
    // flush the data to disk
    msync(data, current_max_size, MS_SYNC);

    // unmap the file out of memory
    munmap(data, consumed_size);

    // resize the file back to the size that it really takes
    if (ftruncate(file_descriptor, consumed_size)) {
      std::cout << "resizing the file back to the correct size did not work.\n";
      close(file_descriptor);
      exit(1);
    }
    close(file_descriptor);
  }
  // const char *begin() const { return data; }
  // const char *end() const { return data + actual_file_size; }

  void write(const void *source, size_t size) {
    if (consumed_size + size > current_max_size) {
      size_t new_max_size = (current_max_size * 2 > consumed_size + size)
                                ? current_max_size * 2
                                : current_max_size * 2 + size;
      resize(new_max_size);
    }
    memcpy(this->data + this->consumed_size, source, size);
    this->consumed_size += size;
  }

 private:
  size_t current_max_size;
  size_t consumed_size = 0;
  int file_descriptor;
  char *data;

  void resize(size_t new_max_size) {
    // flush the data to disk
    msync(data, current_max_size, MS_SYNC);
    munmap(data, consumed_size);
    if (ftruncate(file_descriptor, new_max_size)) {
      std::cout << "resizing the file to " << new_max_size << " did not work.\n";
      close(file_descriptor);
      exit(1);
    }
    current_max_size = new_max_size;
    data = static_cast<char *>(
        mmap(nullptr, current_max_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor, 0));
    std::cout << "successfully resized to " << current_max_size << ".\n";
  }
};

class MmapedFileReader {
 public:
  MmapedFileReader(std::string &path) {
    this->file_descriptor = open(path.c_str(), O_RDONLY);

    if (file_descriptor < 0) exit(1);

    lseek(file_descriptor, 0, SEEK_END);
    size = lseek(file_descriptor, 0, SEEK_CUR);
    data = static_cast<char *>(mmap(nullptr, size, PROT_READ, MAP_SHARED, file_descriptor, 0));
  };
  ~MmapedFileReader() {
    munmap(data, size);
    close(file_descriptor);
  }
  const char *begin() const { return data; }
  const char *end() const { return data + size; }
  const size_t get_size() const { return size; }

 private:
  int file_descriptor;
  size_t size;
  char *data;
};