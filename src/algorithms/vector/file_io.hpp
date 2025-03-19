
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

class MmapedFile {
 public:
  MmapedFile(const char *path, size_t intermediate_size) : current_max_size(intermediate_size) {
    // open the file
    this->file_descriptor = open(path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (file_descriptor < 0) exit(1);

    // set file size of the mapped file to intermediate_size
    if (ftruncate(file_descriptor, intermediate_size)) {
      std::cout << "setting the intermediate file size did not work.\n";
      close(file_descriptor);
      exit(1);
    }

    // map the specified file into memory
    data = static_cast<char *>(
        mmap(nullptr, intermediate_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor, 0));
  };

  ~MmapedFile() {
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

  void write(const std::string &s) {
    if (consumed_size + s.length() > current_max_size) {
      size_t new_max_size = current_max_size * 1.3;
      resize(new_max_size);
    }
    memcpy(this->data + this->consumed_size, s.data(), s.length());
    this->consumed_size += s.length();
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